"""Sealing and verification — the core of ProofPrint.

Sealing embeds the Genblaze provenance manifest *inside* the media file itself
(PNG ``iTXt``, JPEG XMP, MP4 ``uuid`` box), so the proof travels with the asset
even when it is emailed, re-uploaded or pulled out of a CMS. That is precisely
the "machine-readable marking" that EU AI Act Article 50 has required of
synthetic media since 2 August 2026.

Verification then answers three *independent* questions, because a single
boolean cannot distinguish the ways provenance breaks:

  1. Is a manifest present at all?          -> unsigned vs. signed
  2. Is the manifest internally consistent?  -> `Manifest.verify_hash()`
     Editing the embedded prompt/model/timestamp changes the canonical hash.
  3. Do the file's bytes still match what we sealed?
     We record ``sha256(sealed bytes)`` in the immutable B2 ledger at mint time.
     Re-hashing the uploaded file and comparing catches a single flipped pixel.

Layer 3 is what makes this more than self-attestation: the reference hash lives
in an Object-Locked B2 object written before the file ever left our hands, so a
forger would have to alter an immutable record to make a fake pass.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from genblaze_core import Manifest
from genblaze_core.media import get_handler, sniff_mime

from . import storage

log = logging.getLogger("proofprint.seal")

# Formats whose handlers can carry an embedded manifest.
EXT_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "video/mp4": "mp4",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_mime(path: Path) -> str:
    return sniff_mime(path) or "application/octet-stream"


def seal_file(source: Path, manifest: Manifest) -> tuple[Path, str, str]:
    """Embed ``manifest`` into ``source``.

    Returns ``(sealed_path, mime, sealed_sha256)``. ``sealed_sha256`` is the
    digest of the file *after* embedding — the reference value that layer 3 of
    verification compares against.
    """
    mime = detect_mime(source)
    handler = get_handler(mime)
    if handler is None:
        raise RuntimeError(f"No Genblaze media handler can embed a manifest into {mime!r}")

    sealed_path = source.with_name(f"{source.stem}.sealed{source.suffix}")
    handler.embed(source, manifest, sealed_path)

    return sealed_path, mime, sha256_file(sealed_path)


def _manifest_summary(manifest: Manifest) -> dict[str, Any]:
    """Flatten a manifest into what the certificate page renders."""
    run = manifest.run
    steps: list[dict[str, Any]] = []

    for index, step in enumerate(run.steps or []):
        assets = []
        for asset in step.assets or []:
            assets.append(
                {
                    "asset_id": asset.asset_id,
                    "sha256": asset.sha256,
                    "media_type": asset.media_type,
                    "size_bytes": asset.size_bytes,
                    "width": asset.width,
                    "height": asset.height,
                }
            )
        steps.append(
            {
                "index": index,
                "provider": getattr(step, "provider", None),
                "model": getattr(step, "model", None),
                "prompt": getattr(step, "prompt", None),
                "status": str(getattr(step, "status", "")),
                "params": getattr(step, "params", None) or {},
                "started_at": getattr(step, "started_at", None),
                "completed_at": getattr(step, "completed_at", None),
                "cost_usd": getattr(step, "cost_usd", None),
                "assets": assets,
            }
        )

    return {
        "run_id": run.run_id,
        "parent_run_id": run.parent_run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "name": run.name,
        "status": str(run.status),
        "created_at": run.created_at,
        "completed_at": run.completed_at,
        "canonical_hash": manifest.canonical_hash,
        "manifest_uri": manifest.manifest_uri,
        "schema_version": manifest.schema_version,
        "steps": steps,
    }


def verify_bytes(data: bytes, filename: str) -> dict[str, Any]:
    """Run the full three-layer check over an uploaded file."""
    work = storage.settings.work_dir / "verify"
    work.mkdir(parents=True, exist_ok=True)

    uploaded_sha = sha256_bytes(data)
    suffix = Path(filename).suffix or ".bin"
    scratch = work / f"{uploaded_sha[:16]}{suffix}"
    scratch.write_bytes(data)

    result: dict[str, Any] = {
        "filename": filename,
        "uploaded_sha256": uploaded_sha,
        "size_bytes": len(data),
        "mime": detect_mime(scratch),
        "verdict": "UNSIGNED",
        "headline": "No provenance found",
        "detail": "This file carries no Genblaze manifest. Its origin cannot be established.",
        "checks": [],
        "manifest": None,
        "ledger": None,
    }

    def add_check(name: str, passed: bool | None, detail: str) -> None:
        result["checks"].append({"name": name, "passed": passed, "detail": detail})

    # --- Layer 1: is a manifest embedded? ---------------------------------
    handler = get_handler(result["mime"])
    if handler is None:
        add_check(
            "Manifest present",
            False,
            f"{result['mime']} cannot carry an embedded manifest.",
        )
        return result

    try:
        manifest = handler.extract(scratch)
    except Exception as exc:
        add_check("Manifest present", False, f"No embedded manifest ({type(exc).__name__}).")
        return result

    add_check("Manifest present", True, "Genblaze manifest extracted from the file itself.")
    summary = _manifest_summary(manifest)
    result["manifest"] = summary

    # --- Layer 2: is the manifest internally consistent? ------------------
    try:
        hash_ok = bool(manifest.verify_hash())
    except Exception:
        hash_ok = False
    add_check(
        "Manifest integrity (SHA-256 canonical hash)",
        hash_ok,
        (
            "The recomputed canonical hash matches the one recorded at generation "
            "time — prompt, model, parameters and timestamps are unaltered."
            if hash_ok
            else "Canonical hash mismatch: the provenance record has been edited."
        ),
    )

    try:
        report = manifest.verification_report()
        assets_ok = bool(getattr(report, "ok", manifest.verify()))
    except Exception:
        assets_ok = False
    add_check(
        "Declared output integrity",
        assets_ok,
        (
            "Every declared output carries a well-formed SHA-256 digest."
            if assets_ok
            else "One or more declared outputs are missing a valid SHA-256 digest."
        ),
    )

    if not hash_ok:
        result["verdict"] = "TAMPERED"
        result["headline"] = "Provenance record has been altered"
        result["detail"] = (
            "A manifest is embedded, but its canonical hash no longer matches its "
            "contents. Someone edited the recorded prompt, model or parameters."
        )
        return result

    # --- Layer 3: do the bytes match the immutable B2 reference? ----------
    ledger_record = None
    try:
        ledger_record = storage.find_by_sha(uploaded_sha)
    except Exception:
        log.warning("ledger lookup failed", exc_info=True)
        add_check("B2 ledger cross-check", None, "Ledger unreachable; skipped.")

    if ledger_record:
        result["ledger"] = ledger_record
        add_check(
            "B2 ledger cross-check",
            True,
            (
                "Byte-for-byte match against the Object-Locked record written to "
                f"Backblaze B2 at mint time ({ledger_record.get('created_at')})."
            ),
        )
        result["verdict"] = "AUTHENTIC"
        result["headline"] = "Authentic and unmodified"
        result["detail"] = (
            "This file is byte-identical to the asset ProofPrint sealed, and its "
            "provenance record is intact."
        )
        return result

    # Manifest is valid, but these exact bytes are not the ones we sealed.
    known = None
    try:
        known = storage.find_by_run(summary["run_id"])
    except Exception:
        log.warning("run lookup failed", exc_info=True)

    if known:
        result["ledger"] = known
        add_check(
            "B2 ledger cross-check",
            False,
            (
                "The run is on record in B2, but the sealed asset for it hashes to "
                f"{known.get('sealed_sha256', '?')[:16]}…, not {uploaded_sha[:16]}…. "
                "The file has been modified since it was sealed."
            ),
        )
        result["verdict"] = "MODIFIED"
        result["headline"] = "Modified after sealing"
        result["detail"] = (
            "The provenance record is genuine and untampered, but the image bytes no "
            "longer match what was sealed — the file has been re-encoded, cropped or "
            "edited since it left the pipeline."
        )
        return result

    add_check(
        "B2 ledger cross-check",
        False,
        "No matching record in this ProofPrint archive.",
    )
    result["verdict"] = "UNKNOWN_ORIGIN"
    result["headline"] = "Valid manifest, unknown archive"
    result["detail"] = (
        "The embedded manifest is internally consistent, but it was not minted by "
        "this ProofPrint instance, so its bytes cannot be checked against an "
        "immutable reference."
    )
    return result
