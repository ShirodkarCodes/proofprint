"""Metadata forensics for arbitrary images.

ProofPrint's manifest and C2PA both answer "who made this, provably". Neither
says anything about a file that carries neither — which is most files. This
module reads whatever the file does carry: EXIF from the camera, XMP edit
history from the editor, and the text chunks that image generators leave behind.

None of it is trustworthy in the cryptographic sense — EXIF is trivially forged
and trivially stripped — so it is reported as *observations*, never as proof.
The distinction matters, and the UI keeps it visible.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("proofprint.forensics")

# Software strings that indicate a generator rather than a camera.
AI_SOFTWARE_HINTS = (
    "stable diffusion", "automatic1111", "comfyui", "invokeai", "fooocus",
    "midjourney", "dall-e", "dalle", "openai", "firefly", "imagen", "flux",
    "novelai", "leonardo", "ideogram", "gemini", "grok", "nano banana",
)

# PNG/JPEG text keys used by generators to record their parameters.
GENERATOR_KEYS = ("parameters", "prompt", "workflow", "sd-metadata", "Comment", "Dream")

# EXIF tags worth surfacing, grouped for display.
CAMERA_TAGS = ("Make", "Model", "LensModel", "LensMake", "BodySerialNumber", "OwnerName")
CAPTURE_TAGS = (
    "DateTimeOriginal", "ExposureTime", "FNumber", "ISOSpeedRatings",
    "FocalLength", "Flash", "WhiteBalance", "ExposureProgram",
)
EDIT_TAGS = ("Software", "ProcessingSoftware", "DateTime", "HostComputer", "ModifyDate")
RIGHTS_TAGS = ("Artist", "Copyright", "ImageDescription", "XPAuthor", "XPComment")


def _clean(value: Any) -> Any:
    """EXIF values arrive as bytes, IFDRational, tuples — normalise for JSON."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-16-le" if b"\x00" in value[:4] else "utf-8", "replace").strip("\x00 ")
        except Exception:
            return value.hex()[:64]
    if isinstance(value, tuple):
        return [_clean(v) for v in value]
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        try:
            return round(float(value), 6)
        except Exception:
            return str(value)
    if isinstance(value, str):
        return value.strip("\x00 ").strip() or None
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:300]


def _gps(exif_gps: dict[Any, Any]) -> dict[str, Any] | None:
    """Convert EXIF GPS IFD into decimal degrees."""
    from PIL.ExifTags import GPSTAGS

    tags = {GPSTAGS.get(k, k): v for k, v in exif_gps.items()}

    def to_degrees(dms: Any, ref: Any) -> float | None:
        try:
            d, m, s = (float(x) for x in dms)
            value = d + m / 60 + s / 3600
            if str(ref).upper() in ("S", "W"):
                value = -value
            return round(value, 6)
        except Exception:
            return None

    lat = to_degrees(tags.get("GPSLatitude"), tags.get("GPSLatitudeRef"))
    lon = to_degrees(tags.get("GPSLongitude"), tags.get("GPSLongitudeRef"))
    if lat is None or lon is None:
        return None

    out: dict[str, Any] = {"latitude": lat, "longitude": lon}
    if tags.get("GPSAltitude") is not None:
        try:
            out["altitude_m"] = round(float(tags["GPSAltitude"]), 1)
        except Exception:
            pass
    if tags.get("GPSDateStamp"):
        out["gps_timestamp"] = _clean(tags["GPSDateStamp"])
    out["maps_url"] = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=15/{lat}/{lon}"
    return out


def inspect(path: Path) -> dict[str, Any]:
    """Collect every observation available from the file's own metadata."""
    report: dict[str, Any] = {
        "camera": {},
        "capture": {},
        "edits": {},
        "rights": {},
        "gps": None,
        "generator": None,
        "generator_params": None,
        "dimensions": None,
        "format": None,
        "has_exif": False,
        "notes": [],
    }

    try:
        from PIL import Image, ExifTags
    except Exception:
        report["notes"].append("Pillow unavailable; no metadata read.")
        return report

    try:
        with Image.open(path) as image:
            report["format"] = image.format
            report["dimensions"] = f"{image.width}x{image.height}"
            report["mode"] = image.mode

            # --- PNG/text chunks: where generators stash their parameters ---
            text_chunks: dict[str, str] = {}
            for attr in ("text", "info"):
                blob = getattr(image, attr, None)
                if isinstance(blob, dict):
                    for key, value in blob.items():
                        if isinstance(value, str) and value.strip() and len(value) < 8000:
                            text_chunks[str(key)] = value

            for key in GENERATOR_KEYS:
                if key in text_chunks:
                    report["generator_params"] = {"key": key, "value": text_chunks[key][:1500]}
                    break

            # --- EXIF ---
            try:
                exif = image.getexif()
            except Exception:
                exif = None

            if exif:
                report["has_exif"] = True
                flat: dict[str, Any] = {}
                for tag_id, value in exif.items():
                    flat[ExifTags.TAGS.get(tag_id, str(tag_id))] = _clean(value)

                try:
                    for tag_id, value in exif.get_ifd(ExifTags.IFD.Exif).items():
                        flat[ExifTags.TAGS.get(tag_id, str(tag_id))] = _clean(value)
                except Exception:
                    pass

                for group, tags in (
                    ("camera", CAMERA_TAGS),
                    ("capture", CAPTURE_TAGS),
                    ("edits", EDIT_TAGS),
                    ("rights", RIGHTS_TAGS),
                ):
                    for tag in tags:
                        if flat.get(tag) not in (None, "", []):
                            report[group][tag] = flat[tag]

                try:
                    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
                    if gps_ifd:
                        report["gps"] = _gps(gps_ifd)
                except Exception:
                    pass

            # --- XMP: editors record their history here ---
            try:
                xmp = image.getxmp() if hasattr(image, "getxmp") else None
                if xmp:
                    agents = _xmp_history(xmp)
                    if agents:
                        report["edits"]["XMP History"] = agents
            except Exception:
                pass

    except Exception as exc:
        report["notes"].append(f"Could not parse image metadata ({type(exc).__name__}).")
        return report

    # --- generator inference ------------------------------------------------
    haystack = " ".join(
        str(v).lower()
        for v in list(report["edits"].values()) + list(report["camera"].values())
    )
    if report["generator_params"]:
        haystack += " " + report["generator_params"]["value"][:400].lower()

    for hint in AI_SOFTWARE_HINTS:
        if hint in haystack:
            report["generator"] = hint
            break

    # --- plain-language observations ---------------------------------------
    notes = report["notes"]
    if report["generator"]:
        notes.append(
            f"Metadata mentions {report['generator']!r}, which suggests AI generation. "
            "Metadata is self-reported and easily forged — treat as a lead, not proof."
        )
    if report["camera"].get("Make") or report["camera"].get("Model"):
        notes.append("Carries camera EXIF, which is typical of a photograph rather than a generated image.")
    if report["edits"].get("Software"):
        notes.append(f"Last written by {report['edits']['Software']!r}.")
    if report["gps"]:
        notes.append("Contains GPS coordinates — check before publishing, this can deanonymise a subject.")
    if not report["has_exif"] and not report["generator_params"]:
        notes.append(
            "No EXIF and no generator metadata. Common for AI output, screenshots, and "
            "anything re-encoded or uploaded through a platform that strips metadata."
        )

    return report


def _xmp_history(xmp: Any) -> list[str]:
    """Pull softwareAgent entries out of an XMP dict of arbitrary depth."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if len(found) >= 6:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                if "softwareagent" in str(key).lower() and isinstance(value, str):
                    if value not in found:
                        found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(xmp)
    return found
