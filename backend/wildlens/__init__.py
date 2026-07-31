"""Digital Safari Tour Guide — LangGraph backend."""

from .state import MIN_CONFIDENCE, SUMMARY_THRESHOLD, WildlensState

__all__ = [
    "WildlensState",
    "MIN_CONFIDENCE",
    "SUMMARY_THRESHOLD",
    "init_rag",
    "build_graph",
    "make_turn_input",
]


def __getattr__(name: str):
    if name == "init_rag":
        from .rag import init_rag
        return init_rag
    if name in ("build_graph", "make_turn_input"):
        from .graphs import build_graph, make_turn_input
        return {"build_graph": build_graph, "make_turn_input": make_turn_input}[name]
    raise AttributeError(f"module 'safari_guide' has no attribute {name!r}")
