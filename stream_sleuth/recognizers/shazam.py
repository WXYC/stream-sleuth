"""Shazam, through the unofficial ``shazamio`` client with its defaults."""

import asyncio

from shazamio import Shazam

from .base import Identification


class ShazamRecognizer:
    def recognize(self, wav_path: str) -> Identification | None:
        return parse(asyncio.run(_recognize(wav_path)))


def parse(out):
    """Pull artist/song/album/label out of Shazam's response, or None on no match."""
    track = out.get("track") if isinstance(out, dict) else None
    if not track:
        return None
    result = {
        "artist": track.get("subtitle", "") or "",
        "song": track.get("title", "") or "",
        "album": "",
        "label": "",
    }
    for section in track.get("sections", []) or []:
        for md in section.get("metadata", []) or []:
            key = (md.get("title") or "").strip().lower()
            val = md.get("text", "") or ""
            if key == "album" and not result["album"]:
                result["album"] = val
            elif key == "label" and not result["label"]:
                result["label"] = val
    return result


async def _recognize(path):
    return await Shazam().recognize(path)
