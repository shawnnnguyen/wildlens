"""
Text-to-speech utilities.

Engine priority:
  1. edge-tts  — Microsoft Neural voices (free, high quality, async)
  2. gTTS      — Google TTS (simpler, free, requires internet)
  3. no-op     — logs a warning; returns sentinel string

Swap this module's synthesise_audio() for an ElevenLabs implementation
without touching any graph node.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import tempfile

log = logging.getLogger("safari_guide.tts")

# ── Engine availability detection (at import time) ───────────────────────────
_EDGE_TTS_AVAILABLE = False
_GTTS_AVAILABLE = False
_gTTS = None  # bound at module level so static analysers never see it as unbound

try:
    import edge_tts as _edge_tts_module
    _EDGE_TTS_AVAILABLE = True
except ImportError:
    pass

try:
    from gtts import gTTS as _gTTS
    _GTTS_AVAILABLE = True
except ImportError:
    pass


# ── edge-tts async worker ─────────────────────────────────────────────────────

async def _edge_tts_coroutine(text: str, path: str, voice: str) -> None:
    communicator = _edge_tts_module.Communicate(text, voice)
    await communicator.save(path)


def _run_in_thread(coro) -> None:
    """
    Run an async coroutine from a synchronous context by executing it in a
    worker thread with its own event loop.

    Using asyncio.run() inside a fresh thread is safe because new threads
    have no event loop — this avoids the 'This event loop is already running'
    RuntimeError that would occur when called from FastAPI or Jupyter.
    """
    def _worker():
        asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_worker).result()


# ── Public API ────────────────────────────────────────────────────────────────

def synthesise_audio(
    script: str,
    edge_tts_voice: str = "en-US-GuyNeural",
) -> str:
    """
    Convert *script* to an MP3 file in the OS temp directory and return its path.

    The file is owned by the OS and will be cleaned up on reboot (or earlier
    by the OS temp-file janitor). Callers that need persistence should copy
    the file to permanent storage before the process exits.

    Returns "NO_TTS_ENGINE_INSTALLED" if neither engine is available —
    the graph continues cleanly and the caller can handle the sentinel.
    """
    fd, file_path = tempfile.mkstemp(suffix=".mp3", prefix="safari_")
    os.close(fd)  # release the fd; the TTS engine opens the file independently

    if _EDGE_TTS_AVAILABLE:
        log.info("[TTS] edge-tts voice=%s → %s", edge_tts_voice, file_path)
        _run_in_thread(_edge_tts_coroutine(script, file_path, edge_tts_voice))

    elif _GTTS_AVAILABLE:
        log.info("[TTS] gTTS → %s", file_path)
        _gTTS(text=script, lang="en", slow=False).save(file_path)

    else:
        os.unlink(file_path)  # clean up the empty temp file we just created
        log.warning(
            "No TTS engine found. "
            "Install edge-tts (pip install edge-tts) or gTTS (pip install gTTS)."
        )
        return "NO_TTS_ENGINE_INSTALLED"

    return file_path
