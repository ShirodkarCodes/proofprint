"""Offline proof of the ProofPrint core loop: embed -> extract -> verify -> tamper.

No API keys, no network. This exercises exactly the code paths app/seal.py uses.
"""
import hashlib
import struct
import sys
import zlib
from pathlib import Path

from genblaze_core import Asset, Manifest, Run
from genblaze_core.models import Step
from genblaze_core.media import get_handler, sniff_mime

WORK = Path("/tmp/proofprint-test")
WORK.mkdir(parents=True, exist_ok=True)


def make_png(path: Path, colour=(30, 160, 120)) -> Path:
    """Minimal valid 64x64 PNG, no external deps."""
    w = h = 64
    raw = b"".join(b"\x00" + bytes(colour) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return path


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


ok = True


def check(label: str, passed: bool, extra: str = "") -> None:
    global ok
    ok = ok and passed
    print(f"{'✓' if passed else '✗'} {label}" + (f"  — {extra}" if extra else ""))


# --- build a realistic manifest ------------------------------------------
src = make_png(WORK / "raw.png")
raw_sha = sha(src)

asset = Asset(
    url=src.as_uri(),
    media_type="image/png",
    sha256=raw_sha,
    size_bytes=src.stat().st_size,
    width=64,
    height=64,
)
step = Step(
    provider="nvidia",
    model="black-forest-labs/flux.1-schnell",
    prompt="a lighthouse on a basalt cliff during a winter storm",
    assets=[asset],
    params={"aspect_ratio": "1:1"},
)
run = Run(name="proofprint-mint", tenant_id="public-demo", steps=[step])
manifest = Manifest.from_run(run)

check("Manifest built", manifest is not None, f"canonical_hash={str(manifest.canonical_hash)[:16]}…")
check("Fresh manifest verify_hash()", bool(manifest.verify_hash()))
check("Fresh manifest verify()", bool(manifest.verify()))

# --- embed ----------------------------------------------------------------
mime = sniff_mime(src) or "?"
check("MIME sniffed", mime == "image/png", mime)

handler = get_handler(mime)
check("Handler resolved", handler is not None, type(handler).__name__ if handler else "None")

sealed = WORK / "sealed.png"
handler.embed(src, manifest, sealed)
check("Embed produced a file", sealed.exists() and sealed.stat().st_size > 0,
      f"{sealed.stat().st_size}B vs raw {src.stat().st_size}B")

sealed_sha = sha(sealed)
check("Sealed bytes differ from raw", sealed_sha != raw_sha)

# --- extract --------------------------------------------------------------
recovered = handler.extract(sealed)
check("Extract returned a manifest", recovered is not None)
check("Canonical hash survives round-trip",
      recovered.canonical_hash == manifest.canonical_hash,
      f"{str(recovered.canonical_hash)[:16]}…")
check("Recovered manifest verify_hash()", bool(recovered.verify_hash()))
check("Prompt survives round-trip",
      recovered.run.steps[0].prompt == step.prompt,
      repr(recovered.run.steps[0].prompt)[:60])
check("Model survives round-trip",
      recovered.run.steps[0].model == step.model)
check("Asset sha256 survives round-trip",
      recovered.run.steps[0].assets[0].sha256 == raw_sha)

# --- tamper: flip a pixel -------------------------------------------------
tampered = WORK / "tampered.png"
make_png(tampered, colour=(30, 160, 121))   # one channel, one step different
handler.embed(tampered, manifest, tampered)
check("Tampered file hashes differently", sha(tampered) != sealed_sha,
      "layer 3 catches byte edits")

# --- tamper: edit the embedded record ------------------------------------
evil = recovered.model_copy(deep=True)
evil.run.steps[0].prompt = "a photograph of a real lighthouse, shot on Leica"
check("Edited manifest fails verify_hash()", not bool(evil.verify_hash()),
      "layer 2 catches record edits")

print()
print("PASS — core loop is sound" if ok else "FAIL — see above")
sys.exit(0 if ok else 1)
