"""A Genblaze connector for Pollinations — keyless open-weight image generation.

Genblaze ships adapters for eleven providers, and every one of them needs an API
key. That is a real gap for a provenance tool: the whole point of ProofPrint's
cross-provider failover is that it still produces an asset when a provider is
down, and a failover leg that needs its own paid credentials is not a failover
anyone can rely on.

Pollinations serves open-weight image models over a plain HTTP GET with no key
and no account, which makes it an ideal last-resort leg. Implementing it against
``SyncProvider`` — the base for providers whose API returns the artefact in the
same call — takes one method.

The connector is written to the public Genblaze provider contract, so it drops
into a ``Pipeline.step()`` exactly like the first-party adapters and its output
lands in the same provenance manifest.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import httpx
from genblaze_core import Asset
from genblaze_core.providers import ModelRegistry, ModelSpec, SyncProvider

log = logging.getLogger("proofprint.providers.pollinations")

ENDPOINT = "https://image.pollinations.ai/prompt/{prompt}"

# Pollinations routes by model name in a query parameter. These are the
# open-weight families it exposes; unknown names fall through to its default.
MODELS = ("flux", "sana", "turbo")

DEFAULT_PARAMS: dict[str, Any] = {
    "width": 1024,
    "height": 1024,
    "nologo": "true",
    "private": "true",
    "safe": "true",
}


class PollinationsImageProvider(SyncProvider):
    """Keyless text-to-image via Pollinations.

    Implements the Genblaze provider contract: ``generate()`` populates
    ``step.assets`` and the base class wraps it into the submit/poll/fetch
    lifecycle the Pipeline drives.
    """

    name = "pollinations"

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        http_timeout: float = 120.0,
        endpoint: str = ENDPOINT,
        models: ModelRegistry | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(models=models or self.models_default(), **kwargs)
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.http_timeout = http_timeout
        self.endpoint = endpoint

    @staticmethod
    def models_default() -> ModelRegistry:
        registry = ModelRegistry()
        for model_id in MODELS:
            # No pricing: the service is free, so cost_usd stays None rather
            # than being asserted as zero.
            registry.register(ModelSpec(model_id=model_id))
        return registry

    def generate(self, step: Any, config: Any = None) -> Any:
        prompt = (step.prompt or "").strip()
        if not prompt:
            raise ValueError("Pollinations requires a non-empty prompt")

        params: dict[str, Any] = {**DEFAULT_PARAMS}
        for key, value in (step.params or {}).items():
            if key in ("width", "height", "seed", "model", "nologo", "private", "safe"):
                params[key] = value
        params["model"] = step.model or MODELS[0]

        url = self.endpoint.format(prompt=urllib.parse.quote(prompt[:1800], safe=""))

        log.info("pollinations generate model=%s", params["model"])
        with httpx.Client(timeout=self.http_timeout, follow_redirects=True) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.content

        if not payload:
            raise RuntimeError("Pollinations returned an empty body")

        media_type = response.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        if not media_type.startswith("image/"):
            raise RuntimeError(f"Pollinations returned {media_type!r}, not an image")

        suffix = {"image/png": ".png", "image/webp": ".webp"}.get(media_type, ".jpg")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"pollinations-{uuid.uuid4().hex[:12]}{suffix}"
        path.write_bytes(payload)

        width = height = None
        try:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.width, image.height
        except Exception:
            log.debug("could not read dimensions from pollinations output")

        step.assets.append(
            Asset(
                url=path.as_uri(),
                media_type=media_type,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                width=width,
                height=height,
            )
        )
        return step
