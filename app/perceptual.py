"""Perceptual hashing — recognising an image after its bytes are destroyed.

SHA-256 answers "are these the same bytes". It is the right question for
tamper-evidence and the wrong one for the most common thing that actually
happens to a picture: someone screenshots it, a CMS re-encodes it, WhatsApp
strips the metadata and drops it to 60% quality. Every byte changes, the
embedded manifest is gone, and a hash-only verifier can say nothing at all.

A difference hash (dHash) survives all of that. Downscale to 9x8 greyscale and
record whether each pixel is brighter than the one to its right: 64 bits that
track composition rather than encoding. Re-compression, resizing, mild colour
shifts and format changes leave it nearly unchanged, so an unsigned upload can
still be matched against the sealed originals in the B2 ledger.

This is a *lead*, not proof — perceptual hashes have false positives, and the
UI is explicit that a match means "looks like", never "is". But going from
"unknown file" to "this is a re-encode of run 7525ca3f, sealed at 17:02" is the
difference between a verifier that only understands its own pristine output and
one that is useful on the messy files people actually have.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("proofprint.perceptual")

HASH_SIZE = 8  # 8x8 comparisons -> 64-bit hash

# Hamming distance thresholds over 64 bits, from manual calibration against
# re-encodes, resizes and crops of our own sealed output.
NEAR_IDENTICAL = 6   # re-encode / resize / quality change
SIMILAR = 12         # crop, mild edit, overlay


def dhash(path: Path) -> str | None:
    """64-bit difference hash as hex, or None if the image cannot be read."""
    try:
        from PIL import Image

        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except Exception:
            pass

        with Image.open(path) as image:
            # LANCZOS downscale first so JPEG blocking does not drive the bits.
            small = image.convert("L").resize(
                (HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS
            )
            pixels = list(small.getdata())

        bits = 0
        for row in range(HASH_SIZE):
            offset = row * (HASH_SIZE + 1)
            for col in range(HASH_SIZE):
                bits <<= 1
                if pixels[offset + col] > pixels[offset + col + 1]:
                    bits |= 1
        return f"{bits:016x}"
    except Exception:
        log.debug("dhash failed for %s", path, exc_info=True)
        return None


def distance(left: str | None, right: str | None) -> int | None:
    """Hamming distance between two hex dhashes."""
    if not left or not right:
        return None
    try:
        return bin(int(left, 16) ^ int(right, 16)).count("1")
    except ValueError:
        return None


def describe(dist: int) -> tuple[str, str]:
    """Map a distance to a confidence label and a plain-language explanation."""
    if dist == 0:
        return "identical", "Pixel-for-pixel identical composition."
    if dist <= NEAR_IDENTICAL:
        return (
            "near-identical",
            "Almost certainly the same image, re-encoded, resized or saved at a "
            "different quality.",
        )
    if dist <= SIMILAR:
        return (
            "similar",
            "Closely related — consistent with a crop, an overlay or a light edit "
            "of the same image.",
        )
    return "unrelated", "Not a match."


def find_match(candidate: str | None, records: list[dict]) -> dict | None:
    """Best perceptual match for ``candidate`` among ledger records, above the
    SIMILAR threshold. Records carry ``dhash`` from the moment they were
    sealed; older records written before perceptual hashing existed simply
    have none and are skipped, so this degrades quietly on historical data.
    """
    if not candidate:
        return None

    best: dict | None = None
    best_distance = 65

    for record in records:
        dist = distance(candidate, record.get("dhash"))
        if dist is None or dist > SIMILAR:
            continue
        if dist < best_distance:
            best_distance, best = dist, record

    if best is None:
        return None

    confidence, explanation = describe(best_distance)
    return {
        "record": best,
        "distance": best_distance,
        "bits_compared": 64,
        "confidence": confidence,
        "explanation": explanation,
    }


def nearest(candidate: str | None, records: list[dict]) -> int | None:
    """Smallest Hamming distance to any hashed record, with no threshold cutoff.

    Used only to report "we checked, closest was N/64 bits" when nothing
    passed the SIMILAR threshold — so a miss is visibly a miss, not silence
    indistinguishable from never having run the comparison at all.
    """
    if not candidate:
        return None
    best: int | None = None
    for record in records:
        dist = distance(candidate, record.get("dhash"))
        if dist is not None and (best is None or dist < best):
            best = dist
    return best
