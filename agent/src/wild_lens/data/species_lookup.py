"""
Lookup helper over the curated `species_list.json` ground-truth data.

Used to cross-check/canonicalize Gemini's freeform, live `species` output
against the handcrafted species list — see `node_safety_check`'s ground-truth
threat_level escalation and `node_retrieve_information`'s species-filtered
retrieval, both of which need to match Gemini's output (e.g. "African lion
(Panthera leo)") against the canonical `common_name` (e.g. "African Lion")
despite casing/whitespace drift.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("safari_guide.data.species_lookup")

_SPECIES_LIST_PATH = Path(__file__).parent / "species_list.json"

_INDEX: dict[str, dict] | None = None  # lazy singleton, built on first lookup


def _normalize(name: str) -> str:
    """'African lion (Panthera leo)' -> 'african lion'."""
    return " ".join(name.split("(")[0].lower().split())


def _load_index(path: Path | None = None) -> dict[str, dict]:
    try:
        entries = json.loads((path or _SPECIES_LIST_PATH).read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning(f"[species_lookup] Failed to load species_list.json: {exc}")
        return {}
    return {_normalize(entry["common_name"]): entry for entry in entries}


def _get_index() -> dict[str, dict]:
    global _INDEX
    if _INDEX is None:
        _INDEX = _load_index()
    return _INDEX


def lookup_species(raw_species: str) -> dict | None:
    """Return the curated species_list.json entry matching *raw_species*, or None."""
    if not raw_species:
        return None
    return _get_index().get(_normalize(raw_species))


def canonical_common_name(raw_species: str) -> str | None:
    """Return the curated `common_name` for *raw_species*, or None if unlisted."""
    entry = lookup_species(raw_species)
    return entry["common_name"] if entry else None


def ground_truth_threat_level(raw_species: str) -> str | None:
    """Return the curated `threat_level` for *raw_species*, or None if unlisted."""
    entry = lookup_species(raw_species)
    return entry.get("threat_level") if entry else None
