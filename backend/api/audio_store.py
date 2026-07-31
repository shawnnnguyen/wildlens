from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path

log = logging.getLogger("backend.audio_store")

# Configurable via env var; defaults to backend/static/audio/ next to this file
AUDIO_DIR: Path = Path(
    os.getenv("AUDIO_DIR", str(Path(__file__).parent / "static" / "audio"))
)


def ensure_audio_dir() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def store_audio(temp_path: str) -> str:
    """
    Move a TTS temp file into AUDIO_DIR and return the bare filename.

    shutil.move() handles cross-filesystem moves (copy + unlink) transparently,
    so AUDIO_DIR can live on a different partition from the OS temp directory.
    """
    src = Path(temp_path)
    dest = AUDIO_DIR / src.name
    shutil.move(str(src), str(dest))
    return src.name


def resolve_audio_path(filename: str) -> Path:
    return AUDIO_DIR / filename


async def audio_janitor(ttl: int, interval: int) -> None:
    """
    Periodically delete .mp3 files in AUDIO_DIR that are older than *ttl* seconds.

    Runs forever inside the asyncio event loop; cancel the returned task to stop it.
    Sleeps first so a rapid server restart doesn't immediately re-scan.

    Error handling per file:
      FileNotFoundError — already deleted by a concurrent request or prior tick; skip.
      PermissionError   — file is locked (Windows: open file descriptor); skip and
                          retry on the next tick once the client closes the stream.
    """
    log.info("Audio janitor started (ttl=%ds, interval=%ds)", ttl, interval)
    while True:
        await asyncio.sleep(interval)
        cutoff = time.time() - ttl
        deleted = 0
        for f in AUDIO_DIR.glob("*.mp3"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    deleted += 1
            except (FileNotFoundError, PermissionError):
                pass
        if deleted:
            log.info("Audio janitor removed %d expired file(s)", deleted)
