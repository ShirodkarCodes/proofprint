"""Runtime configuration for ProofPrint.

Everything is env-driven so the same image runs locally and on Render.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- Backblaze B2 -----------------------------------------------------
    b2_bucket: str | None = field(default_factory=lambda: _env("B2_BUCKET"))
    b2_region: str | None = field(default_factory=lambda: _env("B2_REGION"))
    b2_key_id: str | None = field(default_factory=lambda: _env("B2_KEY_ID"))
    b2_app_key: str | None = field(default_factory=lambda: _env("B2_APP_KEY"))

    # Prefix that every ProofPrint object lives under inside the bucket.
    b2_prefix: str = field(default_factory=lambda: _env("B2_PREFIX", "proofprint"))

    # Object Lock retention for sealed assets + manifests, in days.
    # GOVERNANCE mode: a key holding s3:BypassGovernanceRetention can still
    # remove the object, so a demo bucket never becomes unrecoverable.
    # Set B2_OBJECT_LOCK_DAYS=0 to disable Object Lock entirely (needed if the
    # bucket was created without Object Lock, which cannot be enabled later).
    object_lock_days: int = field(default_factory=lambda: _env_int("B2_OBJECT_LOCK_DAYS", 1))

    # --- Generative providers --------------------------------------------
    nvidia_api_key: str | None = field(default_factory=lambda: _env("NVIDIA_API_KEY"))
    gemini_api_key: str | None = field(default_factory=lambda: _env("GEMINI_API_KEY"))

    # --- App --------------------------------------------------------------
    tenant_id: str = field(default_factory=lambda: _env("PROOFPRINT_TENANT", "public-demo"))
    work_dir: Path = field(
        default_factory=lambda: Path(_env("PROOFPRINT_WORK_DIR", "/tmp/proofprint"))
    )
    step_timeout: int = field(default_factory=lambda: _env_int("PROOFPRINT_STEP_TIMEOUT", 240))

    @property
    def b2_configured(self) -> bool:
        return bool(self.b2_bucket and self.b2_key_id and self.b2_app_key)

    @property
    def object_lock_enabled(self) -> bool:
        return self.object_lock_days > 0

    def missing(self) -> list[str]:
        """Human-readable list of config the app needs but does not have."""
        gaps: list[str] = []
        if not self.b2_bucket:
            gaps.append("B2_BUCKET")
        if not self.b2_key_id:
            gaps.append("B2_KEY_ID")
        if not self.b2_app_key:
            gaps.append("B2_APP_KEY")
        if not (self.nvidia_api_key or self.gemini_api_key):
            gaps.append("NVIDIA_API_KEY or GEMINI_API_KEY")
        return gaps


settings = Settings()
settings.work_dir.mkdir(parents=True, exist_ok=True)


# --- Model roster ---------------------------------------------------------
# Primary chain runs entirely on open-weight models served by NVIDIA NIM.
# Genblaze handles in-provider model fallback via `fallback_models`; ProofPrint
# adds a cross-provider failover on top (see app/pipeline.py).

NVIDIA_PRIMARY = "black-forest-labs/flux.1-schnell"
NVIDIA_FALLBACKS = [
    "stabilityai/stable-diffusion-3-5-large-turbo",
    "stabilityai/stable-diffusion-xl",
]

GEMINI_PRIMARY = "gemini-2.5-flash-image"
GEMINI_FALLBACKS = ["gemini-3.1-flash-image"]

# Chat model used to expand a short brief into a production prompt.
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
