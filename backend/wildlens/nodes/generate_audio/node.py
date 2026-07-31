"""NODE — Generate audio (conditional on voice_requested)."""
from __future__ import annotations

import logging

from wildlens.state import WildlensState
from wildlens.tts import synthesise_audio

log = logging.getLogger("safari_guide.nodes")


def node_generate_audio(state: WildlensState) -> dict:
    """
    Thin adapter: reads final_script, writes audio_file_path.
    Only reached when voice_requested=True (enforced by route_audio in graphs.py).
    TTS engine swap (e.g. ElevenLabs) requires changes only in tts.py.
    """
    log.info("▶ NODE  generate_audio")
    script = state.get("final_script", "")
    if not script:
        log.warning("   → No script available for TTS.")
        return {"audio_file_path": ""}

    path = synthesise_audio(script)
    log.info("   → Audio saved: %s", path)
    return {"audio_file_path": path}
