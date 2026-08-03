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
import json
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


# An upload is untrusted input, and at least one Genblaze handler (JPEG/XMP) can
# spin indefinitely on real provider output. A worker thread with a deadline
# keeps a hostile or merely awkward file from pinning a request forever.
# 8s is comfortably above the PNG path (~1s on a 3MP still) while keeping the
# JPEG path — whose Genblaze handler can wedge — from stalling a drag-and-drop.
EXTRACT_TIMEOUT_SEC = 8


def _extract_bounded(handler: Any, path: Path) -> Manifest:
    """Extract with a deadline, on a thread that can be safely abandoned.

    Deliberately a raw daemon thread rather than ThreadPoolExecutor: pool
    workers are non-daemon and ``concurrent.futures`` joins them at interpreter
    exit, so one wedged extraction would hang shutdown and leak a thread per
    request. A daemon thread just gets dropped.
    """
    import threading

    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = handler.extract(path)
        except BaseException as exc:  # noqa: BLE001 - relayed to the caller
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True, name="proofprint-extract")
    thread.start()
    thread.join(timeout=EXTRACT_TIMEOUT_SEC)

    if thread.is_alive():
        raise TimeoutError(f"metadata extraction exceeded {EXTRACT_TIMEOUT_SEC}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


# Sealed stills are normalised to PNG before embedding. Two reasons:
#   1. PNG is lossless, so the sealed artefact is pixel-identical to what the
#      model produced — re-encoding a JPEG to carry a manifest would silently
#      degrade the very asset we are certifying.
#   2. Genblaze's JPEG XMP path is not usable here: JpegHandler.extract() hangs
#      on provider-sized JPEGs (reproduced against NVIDIA NIM output), whereas
#      the PNG iTXt path round-trips in milliseconds.
NORMALISE_TO_PNG = {"image/jpeg", "image/webp"}


def _to_png(source: Path) -> Path:
    from PIL import Image

    target = source.with_suffix(".norm.png")
    with Image.open(source) as image:
        image.convert("RGBA" if "A" in image.getbands() else "RGB").save(
            target, format="PNG", optimize=False
        )
    return target


def seal_file(source: Path, manifest: Manifest) -> tuple[Path, str, str]:
    """Embed ``manifest`` into ``source``.

    Returns ``(sealed_path, mime, sealed_sha256)``. ``sealed_sha256`` is the
    digest of the file *after* embedding — the reference value that layer 3 of
    verification compares against. The raw provider output stays in B2 under its
    own content-addressed key, so the manifest's declared asset hash remains
    checkable against the untouched original.
    """
    mime = detect_mime(source)
    if mime in NORMALISE_TO_PNG:
        source = _to_png(source)
        mime = "image/png"

    handler = get_handler(mime)
    if handler is None:
        raise RuntimeError(f"No Genblaze media handler can embed a manifest into {mime!r}")

    sealed_path = source.with_name(f"{source.stem}.sealed{source.suffix}")
    handler.embed(source, manifest, sealed_path)

    return sealed_path, mime, sha256_file(sealed_path)


def read_c2pa(path: Path) -> dict[str, Any] | None:
    """Read C2PA Content Credentials, if this file carries any.

    ProofPrint's own manifests only cover assets it minted. C2PA is the industry
    provenance standard (Adobe, Microsoft, OpenAI, Google; Leica and Sony ship it
    in-camera), so reading it lets the verifier say something useful about an
    image that came from DALL-E or Photoshop rather than shrugging "unsigned".

    Returns None when the file has no Content Credentials, or when the c2pa
    runtime is unavailable — this is strictly an enrichment path and must never
    be able to fail a verification.
    """
    try:
        from c2pa import Reader
    except Exception:
        log.debug("c2pa runtime unavailable")
        return None

    try:
        with Reader(str(path)) as reader:
            payload = json.loads(reader.json())
            try:
                state = str(reader.get_validation_state())
            except Exception:
                state = "unknown"
            try:
                embedded = bool(reader.is_embedded())
            except Exception:
                embedded = True
    except Exception:
        # The overwhelmingly common case: no C2PA data in the file at all.
        return None

    manifests = payload.get("manifests") or {}
    active_id = payload.get("active_manifest")
    active = manifests.get(active_id) if active_id else None
    if active is None and manifests:
        active = next(iter(manifests.values()))
    if active is None:
        return None

    signature = active.get("signature_info") or {}
    actions: list[str] = []
    generators: list[str] = []
    for assertion in active.get("assertions") or []:
        label = assertion.get("label", "")
        data = assertion.get("data") or {}
        if label.startswith("c2pa.actions"):
            for action in data.get("actions") or []:
                name = action.get("action")
                if name:
                    actions.append(name)
                source = action.get("digitalSourceType")
                if source:
                    actions.append(source.rsplit("/", 1)[-1])
        if label.startswith("c2pa.training-mining"):
            actions.append("training-mining-declared")
        soft = data.get("softwareAgent")
        if isinstance(soft, dict) and soft.get("name"):
            generators.append(str(soft["name"]))
        elif isinstance(soft, str):
            generators.append(soft)

    return {
        "claim_generator": active.get("claim_generator_info")
        or active.get("claim_generator")
        or (generators[0] if generators else None),
        "title": active.get("title"),
        "format": active.get("format"),
        "issuer": signature.get("issuer"),
        "signed_at": signature.get("time"),
        "cert_alg": signature.get("alg"),
        "validation_state": state,
        "embedded": embedded,
        "actions": sorted(set(actions))[:8],
        "manifest_count": len(manifests),
    }


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
        "c2pa": None,
        "forensics": None,
    }

    def add_check(name: str, passed: bool | None, detail: str) -> None:
        result["checks"].append({"name": name, "passed": passed, "detail": detail})

    # --- Observations: whatever metadata the file carries ------------------
    # Not evidence — EXIF is forgeable and strippable — but it is often the only
    # thing an unsigned file has, and it answers real questions: what camera,
    # which editor touched it last, where was it taken.
    try:
        from . import forensics

        result["forensics"] = forensics.inspect(scratch)
    except Exception:
        log.warning("forensics pass failed", exc_info=True)
        result["forensics"] = None

    # --- Layer 0: C2PA Content Credentials from any issuer ----------------
    # Runs first and independently of ProofPrint's own manifest, so a file that
    # came from DALL-E, Firefly or a Leica body still gets a useful answer.
    c2pa = read_c2pa(scratch)
    if c2pa:
        result["c2pa"] = c2pa
        valid = "valid" in str(c2pa.get("validation_state", "")).lower()
        issuer = c2pa.get("issuer") or "an undisclosed issuer"
        add_check(
            "C2PA Content Credentials",
            valid or None,
            f"Signed by {issuer}"
            + (f" · {c2pa['claim_generator']}" if c2pa.get("claim_generator") else "")
            + f" · validation: {c2pa.get('validation_state')}",
        )
    else:
        add_check("C2PA Content Credentials", None, "No Content Credentials in this file.")

    def unsigned_outcome() -> dict[str, Any]:
        """No ProofPrint manifest — but C2PA may still establish provenance."""
        if result["c2pa"]:
            result["verdict"] = "EXTERNAL_PROVENANCE"
            result["headline"] = "Provenance from another issuer"
            result["detail"] = (
                "This file was not sealed by ProofPrint, but it carries C2PA Content "
                "Credentials describing how it was made. Those credentials are shown "
                "below; ProofPrint cannot check them against its own immutable ledger."
            )
        return result

    # --- Layer 1: is a ProofPrint manifest embedded? ----------------------
    handler = get_handler(result["mime"])
    if handler is None:
        add_check(
            "Manifest present",
            False,
            f"{result['mime']} cannot carry an embedded Genblaze manifest.",
        )
        return unsigned_outcome()

    try:
        manifest = _extract_bounded(handler, scratch)
    except TimeoutError:
        add_check(
            "Manifest present",
            None,
            f"Timed out reading {result['mime']} metadata after {EXTRACT_TIMEOUT_SEC}s.",
        )
        result["headline"] = "Could not read this file's metadata"
        result["detail"] = (
            "Metadata extraction for this format exceeded the time budget, so no "
            "verdict can be given. Sealed ProofPrint stills are PNG."
        )
        return result
    except Exception as exc:
        add_check("Manifest present", False, f"No embedded manifest ({type(exc).__name__}).")
        return unsigned_outcome()

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
