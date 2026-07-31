"""
LangGraph node implementations for the WildLens.

Every node is a pure function: (state, *deps) -> partial_state_dict.
Dependencies (llm, vectorstore) are injected via closures in graphs.py,
not via module-level globals, so each node is independently testable.

Each node lives in its own submodule alongside its prompts (if any) and its
graph-routing function (the conditional edge that follows it, if any):

    analyze_image/            node_analyze_image           + route_after_analysis
    unclear_photo_fallback/   node_unclear_photo_fallback
    check_relevance/          node_check_relevance          + route_after_relevance + prompts
    topic_redirect_fallback/  node_topic_redirect_fallback
    retrieve_information/     node_retrieve_information
    summarize_history/        node_summarize_history
    generate_guide_persona/   node_generate_guide_persona   + prompts
    generate_audio/           node_generate_audio           + route_audio (shared gate)

This module re-exports the 8 node_* functions so graphs.py's `from .nodes
import (...)` block doesn't need to know about the submodule split.
"""
from __future__ import annotations

from wildlens.nodes.analyze_image.node import node_analyze_image
from wildlens.nodes.check_relevance.node import node_check_relevance
from wildlens.nodes.generate_audio.node import node_generate_audio
from wildlens.nodes.generate_guide_persona.node import node_generate_guide_persona
from wildlens.nodes.retrieve_information.node import node_retrieve_information
from wildlens.nodes.summarize_history.node import node_summarize_history
from wildlens.nodes.topic_redirect_fallback.node import node_topic_redirect_fallback
from wildlens.nodes.unclear_photo_fallback.node import node_unclear_photo_fallback

__all__ = [
    "node_analyze_image",
    "node_check_relevance",
    "node_generate_audio",
    "node_generate_guide_persona",
    "node_retrieve_information",
    "node_summarize_history",
    "node_topic_redirect_fallback",
    "node_unclear_photo_fallback",
]
