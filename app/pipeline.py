"""Genblaze orchestration for ProofPrint.

A mint is a three-tier reliability structure, all of it observable in the
manifest that ends up embedded in the asset:

  tier 1  A Gemini chat step expands a short human brief into a production
          prompt. Recorded in run metadata, so the certificate shows both what
          the human asked for and what the image model was actually given.
  tier 2  Genblaze-native ``fallback_models``: if FLUX.1-dev fails, the same
          provider retries down an open-weight chain without the caller knowing.
  tier 3  Cross-provider failover: if the whole NVIDIA NIM leg is down, the mint
          re-runs against Google. Genblaze's uniform Pipeline API is what makes
          this a provider swap rather than a rewrite.

Every attempt — including the failures — is reported back to the UI, because
"it fell back twice and still delivered" is the actual production story.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genblaze_core import Modality, Pipeline

from . import perceptual, storage
from .config import (
    GEMINI_CHAT_MODELS,
    GEMINI_FALLBACKS,
    GEMINI_PRIMARY,
    NVIDIA_FALLBACKS,
    NVIDIA_PARAMS,
    NVIDIA_PRIMARY,
    POLLINATIONS_FALLBACKS,
    POLLINATIONS_PRIMARY,
    settings,
)

log = logging.getLogger("proofprint.pipeline")

PROMPT_SYSTEM = (
    "You are a prompt engineer for text-to-image models. Rewrite the user's brief "
    "as a single vivid, concrete image prompt. Describe subject, composition, "
    "lighting, lens and art direction. No preamble, no quotes, no markdown, no "
    "line breaks. Under 70 words."
)


# --- tier 1: prompt expansion --------------------------------------------


def expand_prompt(brief: str) -> tuple[str, dict[str, Any]]:
    """Expand a brief via Gemini chat, walking the model list on failure.

    Never fatal: prompt expansion is an enhancement, so any failure degrades to
    generating from the raw brief rather than failing the mint.
    """
    if not settings.gemini_api_key:
        return brief, {"prompt_expansion": "skipped"}

    from genblaze_google import chat

    errors: list[str] = []
    for model in GEMINI_CHAT_MODELS:
        started = time.monotonic()
        try:
            response = chat(
                model,
                prompt=brief,
                system=PROMPT_SYSTEM,
                api_key=settings.gemini_api_key,
                temperature=0.8,
                max_tokens=300,
            )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                errors.append(f"{model}: empty response")
                continue

            return text, {
                "prompt_expansion": "ok",
                "prompt_expansion_model": model,
                "prompt_expansion_provider": "google",
                "prompt_expansion_ms": round((time.monotonic() - started) * 1000),
                "prompt_expansion_attempts": errors or None,
                "original_brief": brief,
            }
        except Exception as exc:
            log.warning("prompt expansion via %s failed: %s", model, exc)
            errors.append(f"{model}: {type(exc).__name__}")

    return brief, {"prompt_expansion": "failed", "prompt_expansion_attempts": errors}


# --- provider legs --------------------------------------------------------


def _providers(only: str | None = None) -> list[dict[str, Any]]:
    """Ordered generation legs, skipping any whose key is absent.

    ``only`` pins the mint to a single provider (by ``id``) instead of walking
    the failover chain — used by the Studio's provider selector so a user can
    compare providers deliberately rather than only seeing whichever one
    happened to answer first.
    """
    legs: list[dict[str, Any]] = []

    if settings.nvidia_api_key:
        from genblaze_nvidia import NvidiaImageProvider

        legs.append(
            {
                "id": "nvidia",
                "name": "NVIDIA NIM",
                # NIM functions cold-start, and the first call after an idle
                # period can sit well past the 120s default before returning.
                "factory": lambda out: NvidiaImageProvider(
                    api_key=settings.nvidia_api_key,
                    output_dir=out,
                    http_timeout=300.0,
                    nvcf_timeout=300.0,
                ),
                "model": NVIDIA_PRIMARY,
                "fallbacks": NVIDIA_FALLBACKS,
                "params": NVIDIA_PARAMS,
            }
        )

    # Keyless, so it is always available — the leg that makes failover real.
    from .providers import PollinationsImageProvider

    legs.append(
        {
            "id": "pollinations",
            "name": "Pollinations",
            "factory": lambda out: PollinationsImageProvider(output_dir=out),
            "model": POLLINATIONS_PRIMARY,
            "fallbacks": POLLINATIONS_FALLBACKS,
            "params": {"width": 1024, "height": 1024},
            "keyless": True,
        }
    )

    if settings.gemini_api_key:
        from genblaze_google import GeminiImageProvider

        legs.append(
            {
                "id": "google",
                "name": "Google Gemini",
                "factory": lambda out: GeminiImageProvider(
                    api_key=settings.gemini_api_key, output_dir=out
                ),
                "model": GEMINI_PRIMARY,
                "fallbacks": GEMINI_FALLBACKS,
                "params": {},
            }
        )

    if only:
        picked = [leg for leg in legs if leg["id"] == only]
        if not picked:
            raise RuntimeError(
                f"Unknown or unavailable provider {only!r}. "
                f"Available: {', '.join(leg['id'] for leg in legs)}"
            )
        return picked

    return legs


def _unwrap(result: Any) -> tuple[Any, Any]:
    """Return (run, manifest) from whatever shape ``Pipeline.run`` handed back."""
    run = getattr(result, "run", None)
    manifest = getattr(result, "manifest", None)
    if run is not None and manifest is not None:
        return run, manifest
    if isinstance(result, tuple) and len(result) == 2:
        return result
    raise RuntimeError(f"Unrecognised PipelineResult shape: {type(result)!r}")


def _first_asset(run: Any) -> Any:
    for step in run.steps or []:
        for asset in step.assets or []:
            return asset
    raise RuntimeError("Pipeline completed but produced no assets")


def _assert_produced(result: Any, run: Any) -> None:
    """Treat a step failure as a leg failure so cross-provider failover fires.

    ``Pipeline.run()`` does not raise by default — it returns a result whose
    steps carry ``status='failed'`` and an ``error``. Without this check a dead
    provider looks like success right up until asset extraction, and the
    failover leg never gets a chance to run.
    """
    # failed_steps is a method on PipelineResult, not a property — calling it is
    # the difference between inspecting failures and truthily inspecting a
    # bound method.
    failed_steps = getattr(result, "failed_steps", None)
    failed = list(failed_steps() or []) if callable(failed_steps) else list(failed_steps or [])
    if failed:
        reasons = "; ".join(
            str(getattr(step, "error", None) or getattr(step, "error_code", "unknown"))
            for step in failed
        )
        raise RuntimeError(reasons or "step failed")

    if not any(step.assets for step in (run.steps or [])):
        raise RuntimeError(f"run finished with status={run.status!s} but produced no assets")


def _materialize(asset: Any, output_dir: Path) -> Path:
    """Get the generated bytes onto local disk so we can embed the manifest.

    Tried in order: the provider's local output_dir, the content-addressed B2
    key, then the asset URL. The local path is preferred because it is the only
    one guaranteed to be the exact bytes the manifest hashed.
    """
    # 1. provider output_dir
    candidates = sorted(
        (p for p in output_dir.glob("**/*") if p.is_file() and p.stat().st_size > 0),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        from .seal import sha256_file

        if asset.sha256 and sha256_file(path) == asset.sha256:
            return path
    if candidates:
        return candidates[0]

    # 2. content-addressed B2 key
    url = str(getattr(asset, "url", "") or "")
    if asset.sha256:
        sha = asset.sha256
        ext = (url.rsplit(".", 1)[-1] if "." in url else "png").split("?")[0][:5] or "png"
        key = f"{settings.b2_prefix}/assets/{sha[:2]}/{sha[2:4]}/{sha}.{ext}"
        try:
            data = storage.get_bytes(key)
            path = output_dir / f"{sha}.{ext}"
            path.write_bytes(data)
            return path
        except Exception:
            log.debug("content-addressed fetch missed for %s", key)

    # 3. plain URL
    if url.startswith("file://"):
        return Path(url[7:])
    if url.startswith("http"):
        import httpx

        response = httpx.get(url, timeout=120, follow_redirects=True)
        response.raise_for_status()
        path = output_dir / f"{uuid.uuid4().hex}.png"
        path.write_bytes(response.content)
        return path

    raise RuntimeError("Could not materialise the generated asset locally")


# --- mint -----------------------------------------------------------------


def mint(
    brief: str,
    *,
    project_id: str = "default",
    parent_run_id: str | None = None,
    expand: bool = True,
    provider: str | None = None,
) -> dict[str, Any]:
    """Generate -> seal -> store -> record. Returns the ledger entry."""
    if not settings.b2_configured:
        raise RuntimeError("Backblaze B2 is not configured.")

    legs = _providers(only=provider)
    if not legs:
        raise RuntimeError("No generation provider configured. Set NVIDIA_API_KEY or GEMINI_API_KEY.")

    prompt, expansion_meta = (expand_prompt(brief) if expand else (brief, {"prompt_expansion": "off"}))

    work = settings.work_dir / f"mint-{uuid.uuid4().hex[:12]}"
    work.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, Any]] = []
    run = manifest = None
    used: dict[str, Any] | None = None

    for leg in legs:
        started = time.monotonic()
        sink = storage.build_sink()
        try:
            pipe = Pipeline(
                "proofprint-mint",
                tenant_id=settings.tenant_id,
                project_id=project_id,
            )
            # Lineage: from_result() only sets this attribute, and we carry the
            # parent across HTTP requests rather than holding results in memory.
            if parent_run_id:
                pipe._parent_run_id = parent_run_id

            # Run-scoped metadata is keyword-only and additive across calls.
            pipe.metadata(
                app="proofprint",
                brief=brief,
                generation_provider=leg["name"],
                **{k: v for k, v in expansion_meta.items() if v is not None},
            )

            pipe.step(
                leg["factory"](work),
                model=leg["model"],
                prompt=prompt,
                modality=Modality.IMAGE,
                fallback_models=leg["fallbacks"],
                **leg.get("params", {}),
            )

            # raise_on_failure=True is the genblaze-core 0.4.0 default; opting in
            # now makes a dead provider raise here instead of returning a
            # success-shaped result with failed steps inside it.
            result = pipe.run(
                sink=sink,
                timeout=settings.step_timeout,
                raise_on_failure=True,
            )
            run, manifest = _unwrap(result)
            _assert_produced(result, run)
            used = leg
            attempts.append(
                {
                    "provider": leg["name"],
                    "model": leg["model"],
                    "status": "ok",
                    "ms": round((time.monotonic() - started) * 1000),
                }
            )
            break
        except Exception as exc:
            log.warning("provider leg %s failed: %s", leg["name"], exc, exc_info=True)
            attempts.append(
                {
                    "provider": leg["name"],
                    "model": leg["model"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                    "ms": round((time.monotonic() - started) * 1000),
                }
            )
        finally:
            try:
                sink.close()
            except Exception:
                pass

    if run is None or manifest is None or used is None:
        raise RuntimeError(
            "All generation providers failed: "
            + "; ".join(f"{a['provider']}: {a.get('error', '?')}" for a in attempts)
        )

    # --- seal: embed the manifest into the file itself --------------------
    from .seal import seal_file, sha256_file

    asset = _first_asset(run)
    raw_path = _materialize(asset, work)
    sealed_path, mime, sealed_sha = seal_file(raw_path, manifest)

    from .seal import EXT_BY_MIME

    ext = EXT_BY_MIME.get(mime, "png")
    key = storage.sealed_key(sealed_sha, ext)
    storage.put_bytes(
        key,
        sealed_path.read_bytes(),
        content_type=mime,
        metadata={
            "run-id": str(run.run_id),
            "canonical-hash": str(manifest.canonical_hash or ""),
            "raw-sha256": str(asset.sha256 or ""),
        },
        immutable=True,
    )

    record: dict[str, Any] = {
        "run_id": str(run.run_id),
        "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
        "tenant_id": settings.tenant_id,
        "project_id": project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "brief": brief,
        "prompt": prompt,
        "provider": used["name"],
        "provider_id": used["id"],
        "model": used["model"],
        "fallback_chain": used["fallbacks"],
        "params": used.get("params", {}),
        "attempts": attempts,
        "expansion": expansion_meta,
        # sha256 is the sealed digest: the identity a verifier presents.
        "sha256": sealed_sha,
        "sealed_sha256": sealed_sha,
        "raw_sha256": asset.sha256,
        # Survives re-encoding, so a stripped copy can still be traced back.
        "dhash": perceptual.dhash(sealed_path),
        "canonical_hash": manifest.canonical_hash,
        "manifest_uri": manifest.manifest_uri,
        "sealed_key": key,
        "mime": mime,
        "size_bytes": sealed_path.stat().st_size,
        # Record what was actually applied, not what was merely requested.
        "object_lock": storage.object_lock_status(),
        "object_lock_days": settings.object_lock_days if storage.probe_object_lock() else 0,
        "width": getattr(asset, "width", None),
        "height": getattr(asset, "height", None),
    }

    storage.write_ledger(record)

    # Drop the whole per-mint working directory, not just the two files we
    # named: the provider's original download and the PNG normalisation step
    # both leave artefacts behind, and a long-running container accumulates
    # them. The durable copies are already in B2.
    import shutil

    shutil.rmtree(work, ignore_errors=True)

    return record
