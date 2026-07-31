"""
Lookup helper over the curated `species_list.json` ground-truth data.

Used to cross-check/canonicalize Gemini's freeform, live `species` output
against the handcrafted species list — see `node_analyze_image`'s ground-truth
threat_level escalation and `node_retrieve_information`'s species-filtered
retrieval, both of which need to match Gemini's output (e.g. "African lion
(Panthera leo)") against the canonical `common_name` (e.g. "African Lion")
despite casing/whitespace drift.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger("safari_guide.data.species_lookup")

_SPECIES_LIST_PATH = Path(__file__).parent / "species_list.json"

_INDEX: dict[str, dict] | None = None  # lazy singleton, built on first lookup

# Head nouns common enough in ordinary English (or, for "python", collides
# with an unrelated common tech term) that a bare word-boundary match would
# false-positive on unrelated text (e.g. "my dog ran away", "check the
# monitor", "python script") — species whose last word falls here get a
# two-word alias instead (see _derive_alias). Determined by inspecting the
# actual species_list.json head-noun set, not a general-purpose stopword list.
_GENERIC_HEAD_NOUNS = {"dog", "ass", "bird", "tree", "monitor", "python", "roller", "monkey"}

# Common informal short forms that don't derive from any common_name's tail
# word at all (e.g. "hippo" isn't a substring or suffix of "Hippopotamus")
# — added by hand since _derive_alias can't discover these algorithmically.
# Merged into the alias index alongside the derived aliases in
# _build_alias_index, including the same collision handling (a candidate
# list, not a single species) where more than one species shares a term.
_MANUAL_ALIASES: dict[str, list[str]] = {
    "hippo": ["Hippopotamus"],
    "rhino": ["White Rhinoceros", "Black Rhinoceros"],
    "croc": ["Nile Crocodile"],
}

_ALIAS_INDEX: dict[str, list[str]] | None = None  # lazy singleton, built on first lookup


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


# ── Free-text species-mention scan (used by node_check_relevance) ─────────────

def _derive_alias(common_name: str) -> str:
    """
    Short informal alias used to find this species mentioned in free text:
    the head noun (last word — "Elephant" from "African Elephant"), unless
    that word alone is common enough in ordinary English to cause false
    positives (see _GENERIC_HEAD_NOUNS), in which case the last two words
    are used instead ("Wild Dog" rather than just "Dog").
    """
    words = common_name.split()
    head = words[-1]
    if head.lower() in _GENERIC_HEAD_NOUNS and len(words) >= 2:
        return " ".join(words[-2:])
    return head


def _build_alias_index() -> dict[str, list[str]]:
    """alias (lowercase) -> [common_name, ...], ordered by first appearance
    in species_list.json, then _MANUAL_ALIASES appended. More than one
    common_name under the same alias (e.g. 'gazelle' -> Thomson's/Grant's
    Gazelle, or 'rhino' -> White/Black Rhinoceros) is a real collision,
    resolved at lookup time in find_mentioned_species — not here."""
    index: dict[str, list[str]] = {}
    for entry in _get_index().values():
        alias = _derive_alias(entry["common_name"]).lower()
        index.setdefault(alias, [])
        if entry["common_name"] not in index[alias]:
            index[alias].append(entry["common_name"])
    for alias, common_names in _MANUAL_ALIASES.items():
        index.setdefault(alias, [])
        for common_name in common_names:
            if common_name not in index[alias]:
                index[alias].append(common_name)
    return index


def _get_alias_index() -> dict[str, list[str]]:
    global _ALIAS_INDEX
    if _ALIAS_INDEX is None:
        _ALIAS_INDEX = _build_alias_index()
    return _ALIAS_INDEX


def _word_boundary_pattern(phrase: str) -> re.Pattern:
    """
    Word-boundary regex for *phrase*, matching the bare form or a naive
    plural (+s/+es) — e.g. 'elephant' matches 'elephant'/'elephants' but not
    'elephantine'. Deliberately not full stemming: a handful of suffix forms
    covers the common case without pulling in a stemming dependency.
    """
    escaped = re.escape(phrase)
    return re.compile(rf"\b{escaped}(?:es|s)?\b", re.IGNORECASE)


def find_mentioned_species(text: str, session_species: list[str] | None = None) -> str | None:
    """
    Scan free text for a curated species name via word-boundary matching
    against aliases derived from species_list.json (see _derive_alias).
    Returns the canonical common_name of the longest matching alias, or None
    if nothing matches.

    Deliberately NOT the same as canonical_common_name()/lookup_species()
    above, which do an exact match on an already-isolated species string
    (e.g. Gemini's structured-output field) — this scans arbitrary free text
    for an embedded mention instead.

    session_species: this session's identification_history species (already
    canonicalized), most recent first. Used to break ties when an alias
    matches more than one curated species (e.g. "gazelle") — prefers
    whichever was most recently identified this session; falls back to the
    first candidate (by species_list.json order) if none were seen.
    """
    if not text:
        return None
    alias_index = _get_alias_index()
    # Longest alias first so a more specific multi-word alias (e.g. "wild dog")
    # wins over any shorter alias that might also appear in the same text.
    for alias in sorted(alias_index, key=len, reverse=True):
        if _word_boundary_pattern(alias).search(text):
            candidates = alias_index[alias]
            if len(candidates) == 1:
                return candidates[0]
            if session_species:
                for seen in session_species:
                    if seen in candidates:
                        return seen
            return candidates[0]
    return None
