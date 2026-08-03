"""ProofPrint HTTP API + single-page frontend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from . import pipeline, seal, storage
from .config import GEMINI_CHAT_MODELS, settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("proofprint")

STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="ProofPrint",
    description="Chain of custody for AI-generated media. Built on Genblaze + Backblaze B2.",
    version="1.0.0",
)


class MintRequest(BaseModel):
    brief: str = Field(min_length=3, max_length=2000)
    project_id: str = "default"
    parent_run_id: str | None = None
    expand: bool = True
    # None = walk the failover chain; an id pins the mint to one provider.
    provider: str | None = None


@app.on_event("startup")
def warm_caches() -> None:
    """Settle Object Lock and prime the ledger off the request path.

    The first ledger read fans out a GET per record and can take a long time on
    a cold or lossy link. Doing it in a background thread at boot means the
    first person to open the app gets the cached copy instead of wearing it.
    """
    import threading

    def warm() -> None:
        try:
            storage.probe_object_lock()
            records = storage.list_ledger(limit=500)
            log.info("ledger cache warmed: %d records", len(records))
        except Exception:
            log.warning("cache warm failed; will populate on first request", exc_info=True)

    if settings.b2_configured:
        threading.Thread(target=warm, daemon=True, name="proofprint-warm").start()


# --- pages ----------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/c/{sha256}", include_in_schema=False)
@app.get("/verify", include_in_schema=False)
@app.get("/ledger", include_in_schema=False)
def spa_routes(sha256: str | None = None) -> FileResponse:
    """Client-side routes all serve the same shell."""
    return FileResponse(STATIC / "index.html")


# --- api ------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Config transparency — judges can see exactly what is wired up."""
    return {
        "status": "ok" if not settings.missing() else "misconfigured",
        "missing_env": settings.missing(),
        "b2": {
            "configured": settings.b2_configured,
            "bucket": settings.b2_bucket,
            "prefix": settings.b2_prefix,
            "key_strategy": "CONTENT_ADDRESSABLE",
            "object_lock": storage.object_lock_status(),
        },
        # Derived from the same list the pipeline walks, so the UI can never
        # advertise a provider the pipeline would not actually use.
        "providers": [
            {
                "id": leg["id"],
                "name": leg["name"],
                "primary": leg["model"],
                "fallbacks": leg["fallbacks"],
                "keyless": leg.get("keyless", False),
            }
            for leg in pipeline._providers()
        ],
        "prompt_expansion": {
            "enabled": bool(settings.gemini_api_key),
            "provider": "google",
            "models": GEMINI_CHAT_MODELS,
        },
    }


@app.post("/api/mint")
def mint(request: MintRequest) -> dict[str, Any]:
    try:
        return pipeline.mint(
            request.brief,
            project_id=request.project_id,
            parent_run_id=request.parent_run_id,
            expand=request.expand,
            provider=request.provider,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("mint failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/verify")
async def verify(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(data) > 64 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File larger than 64 MB")
    try:
        return seal.verify_bytes(data, file.filename or "upload.bin")
    except Exception as exc:
        log.exception("verify failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/ledger")
def ledger(limit: int = 60) -> dict[str, Any]:
    try:
        records = storage.list_ledger(limit=limit)
    except Exception as exc:
        log.exception("ledger read failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"count": len(records), "records": records}


@app.get("/api/record/{sha256}")
def record(sha256: str) -> dict[str, Any]:
    found = storage.find_by_sha(sha256)
    if not found:
        raise HTTPException(status_code=404, detail="No sealed asset with that digest")
    return found


@app.get("/api/lineage/{run_id}")
def lineage(run_id: str) -> dict[str, Any]:
    """Walk parent_run_id backwards, then collect direct children."""
    records = storage.list_ledger(limit=500)
    by_run = {r["run_id"]: r for r in records}

    chain: list[dict[str, Any]] = []
    cursor: str | None = run_id
    seen: set[str] = set()
    while cursor and cursor in by_run and cursor not in seen:
        seen.add(cursor)
        chain.append(by_run[cursor])
        cursor = by_run[cursor].get("parent_run_id")
    chain.reverse()

    children = [r for r in records if r.get("parent_run_id") == run_id]
    return {"ancestors": chain, "children": children}


@app.get("/api/asset/{sha256}")
def asset(sha256: str) -> RedirectResponse:
    """Redirect to a short-lived presigned B2 URL — bytes are served by B2, not us."""
    found = storage.find_by_sha(sha256)
    if not found:
        raise HTTPException(status_code=404, detail="Unknown asset")
    try:
        return RedirectResponse(storage.presign(found["sealed_key"], expires_in=3600))
    except Exception as exc:
        log.exception("presign failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.exception_handler(RuntimeError)
def runtime_error_handler(_request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})
