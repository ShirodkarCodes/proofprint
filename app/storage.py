"""Backblaze B2 layer for ProofPrint.

Bucket layout (everything under ``{prefix}/``)::

    proofprint/
      runs/                      <- written by Genblaze ObjectStorageSink
        {tenant}/{date}/{run_id}/
          manifest.json          <- Object Lock protected
          assets/{asset_id}.png  <- the raw provider output
      sealed/
        {sha256}.png             <- output with the manifest embedded in-file,
                                    content-addressed, Object Lock protected
      ledger/
        {created_at}_{run_id}.json  <- append-only index record, one per mint
      analytics/
        runs.parquet, steps.parquet, assets.parquet  <- ParquetSink tables

The ledger is deliberately one immutable object per run rather than a single
mutable index file: concurrent mints can never clobber each other, and listing
the prefix is a cheap, ordered scan because keys are timestamp-prefixed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from genblaze_core import KeyStrategy, ObjectLockConfig, ObjectStorageSink
from genblaze_s3 import S3StorageBackend

from .config import settings

log = logging.getLogger("proofprint.storage")

_backend: S3StorageBackend | None = None

# Tri-state probe result: None = not yet checked, True/False = usable or not.
_object_lock_usable: bool | None = None


def object_lock() -> ObjectLockConfig | None:
    """Object Lock config for immutable writes, or None if unavailable.

    GOVERNANCE mode keeps the demo bucket recoverable by a privileged key while
    still making ordinary deletes and overwrites fail — which is the property a
    chain-of-custody archive actually needs.

    Returns None when Object Lock is switched off *or* when the bucket/key turns
    out not to support it. See :func:`probe_object_lock`.
    """
    if not settings.object_lock_enabled:
        return None
    if not probe_object_lock():
        return None
    retain_until = datetime.now(timezone.utc) + timedelta(days=settings.object_lock_days)
    return ObjectLockConfig(retain_until=retain_until, mode="GOVERNANCE")


def probe_object_lock() -> bool:
    """Check once whether this bucket + key can actually write retention headers.

    Object Lock only exists on buckets created with it enabled, and the
    application key must carry ``writeFileRetentions``. Rather than let a mint
    die half-way through with an AccessDenied, we settle the question up front
    with one throwaway object and degrade to un-locked writes if it fails.
    """
    global _object_lock_usable
    if _object_lock_usable is not None:
        return _object_lock_usable
    if not settings.object_lock_enabled:
        _object_lock_usable = False
        return False

    key = f"{settings.b2_prefix}/.probe/object-lock"
    retain_until = datetime.now(timezone.utc) + timedelta(days=settings.object_lock_days)
    try:
        backend().put(
            key,
            b"proofprint object-lock probe",
            content_type="text/plain",
            object_lock=ObjectLockConfig(retain_until=retain_until, mode="GOVERNANCE"),
        )
        _object_lock_usable = True
        log.info("Object Lock probe OK — sealed assets will be written immutably")
    except Exception as exc:
        _object_lock_usable = False
        log.warning(
            "Object Lock unavailable (%s: %s) — falling back to standard writes. "
            "The bucket must be created with Object Lock enabled and the app key "
            "must allow writeFileRetentions.",
            type(exc).__name__,
            exc,
        )
    return _object_lock_usable


def object_lock_status() -> str:
    if not settings.object_lock_enabled:
        return "disabled (B2_OBJECT_LOCK_DAYS=0)"
    if not settings.b2_configured:
        return "unknown (B2 not configured)"
    if probe_object_lock():
        return f"GOVERNANCE, {settings.object_lock_days}d"
    return "unavailable (bucket or key lacks Object Lock)"


def backend() -> S3StorageBackend:
    """Lazily build (and memoise) the B2 storage backend."""
    global _backend
    if _backend is None:
        if not settings.b2_configured:
            raise RuntimeError(
                "Backblaze B2 is not configured. Set B2_BUCKET, B2_KEY_ID and B2_APP_KEY."
            )
        _backend = S3StorageBackend.for_backblaze(
            settings.b2_bucket,
            region=settings.b2_region,
            key_id=settings.b2_key_id,
            app_key=settings.b2_app_key,
        )
    return _backend


def build_sink() -> ObjectStorageSink:
    """A fresh Genblaze sink per run.

    CONTENT_ADDRESSABLE keying means two identical outputs collapse onto one
    object — a real dedup win once a team generates thousands of variants — and
    the key *is* the SHA-256, so the storage path itself is a integrity claim.
    """
    return ObjectStorageSink(
        backend(),
        prefix=settings.b2_prefix,
        key_strategy=KeyStrategy.CONTENT_ADDRESSABLE,
        manifest_lock=object_lock(),
    )


# --- Raw object helpers ---------------------------------------------------


def put_bytes(
    key: str,
    data: bytes,
    *,
    content_type: str,
    metadata: dict[str, str] | None = None,
    immutable: bool = False,
) -> str:
    return backend().put(
        key,
        data,
        content_type=content_type,
        metadata=metadata or {},
        object_lock=object_lock() if immutable else None,
    )


def get_bytes(key: str, *, attempts: int = 3) -> bytes:
    """Fetch an object, retrying the mid-stream resets B2 occasionally serves.

    A dropped response body surfaces as StorageError only after the request
    succeeded, so it is not covered by botocore's own retry policy.
    """
    import time as _time

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return backend().get(key)
        except Exception as exc:
            last = exc
            if attempt < attempts - 1:
                _time.sleep(0.4 * (2**attempt))
                log.warning("retrying get for %s after %s", key, type(exc).__name__)
    raise last  # type: ignore[misc]


def exists(key: str) -> bool:
    try:
        return bool(backend().exists(key))
    except Exception:
        return False


def presign(key: str, *, expires_in: int = 3600) -> str:
    """Time-limited read URL so judges can open assets in a private bucket."""
    return backend().presigned_get_url(key, expires_in=expires_in)


# --- Key builders ---------------------------------------------------------


def sealed_key(sha256: str, ext: str = "png") -> str:
    return f"{settings.b2_prefix}/sealed/{sha256}.{ext}"


def ledger_key(created_at: str, run_id: str) -> str:
    safe = created_at.replace(":", "").replace("-", "")
    return f"{settings.b2_prefix}/ledger/{safe}_{run_id}.json"


# --- Ledger ---------------------------------------------------------------


def write_ledger(record: dict[str, Any]) -> str:
    """Write the ledger entry immutably.

    This record holds the reference digest that layer 3 of verification compares
    against, so it is exactly the thing an attacker would need to rewrite. It is
    written under Object Lock for the same reason the sealed asset is.
    """
    key = ledger_key(record["created_at"], record["run_id"])
    put_bytes(
        key,
        json.dumps(record, indent=2, default=str).encode(),
        content_type="application/json",
        metadata={
            "run-id": record["run_id"],
            "sha256": record.get("sha256", ""),
        },
        immutable=True,
    )
    invalidate_ledger_cache()
    return key


# A single verification can consult the ledger three times (exact-hash lookup,
# run lookup, perceptual sweep) and each record is its own B2 object, so an
# uncached scan issues O(records) GETs per request and degrades as the archive
# grows. Ledger objects are immutable and only ever appended, so a short TTL is
# safe: the worst case is a newly minted asset staying invisible for a few
# seconds, and write_ledger drops the cache anyway.
_LEDGER_TTL_SEC = 20.0
_ledger_cache: tuple[float, list[dict[str, Any]]] | None = None


def invalidate_ledger_cache() -> None:
    global _ledger_cache
    _ledger_cache = None


def list_ledger(limit: int = 200) -> list[dict[str, Any]]:
    """Newest-first scan of the ledger prefix, cached briefly."""
    global _ledger_cache

    import time as _time

    if _ledger_cache is not None:
        cached_at, records = _ledger_cache
        if _time.monotonic() - cached_at < _LEDGER_TTL_SEC:
            return records[:limit]

    records = _read_ledger()
    _ledger_cache = (_time.monotonic(), records)
    return records[:limit]


def _read_ledger() -> list[dict[str, Any]]:
    prefix = f"{settings.b2_prefix}/ledger/"
    keys: list[str] = []
    token: str | None = None

    while True:
        page = backend().list(prefix, max_keys=1000, continuation_token=token)
        for entry in page.entries or []:
            keys.append(entry.key if hasattr(entry, "key") else str(entry))
        token = page.next_token
        if not token:
            break

    keys.sort(reverse=True)  # timestamp-prefixed keys sort chronologically
    if not keys:
        return []

    # Each ledger entry is its own object, so a serial scan pays one full B2
    # round trip per record — measured at ~5s each, which made even a handful of
    # records take half a minute. The fetches are independent, so issue them
    # together and keep the wall clock at roughly one round trip.
    from concurrent.futures import ThreadPoolExecutor

    def fetch(key: str) -> dict[str, Any] | None:
        try:
            return json.loads(get_bytes(key))
        except Exception:
            log.warning("skipping unreadable ledger object %s", key, exc_info=True)
            return None

    # Stay under botocore's default connection pool of 10, or the extra workers
    # just queue on the pool and can trip connection resets.
    with ThreadPoolExecutor(max_workers=min(8, len(keys))) as pool:
        fetched = list(pool.map(fetch, keys))

    return [record for record in fetched if record is not None]


def find_by_sha(sha256: str) -> dict[str, Any] | None:
    for record in list_ledger(limit=500):
        if record.get("sha256") == sha256:
            return record
    return None


def find_by_run(run_id: str) -> dict[str, Any] | None:
    for record in list_ledger(limit=500):
        if record.get("run_id") == run_id:
            return record
    return None
