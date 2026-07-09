"""
Unit tests for the golden-dataset eval scorers and task function (Phase 3
observability/eval hardening). Fully mocked — no real Langfuse/LLM calls,
matching this repo's existing test convention (agent/tests/).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langfuse import Evaluation

from wildlens.eval.scorers import (
    ABSTAIN_ERROR,
    EVAL_TASK_ERROR,
    confidence_calibration,
    make_persona_hallucination_evaluator,
    mean_confidence_when_wrong,
    species_match,
    threat_level_match,
)
from wildlens.eval.run_eval import make_task


# ── species_match ──────────────────────────────────────────────────────────

def test_species_match_correct():
    output = {"identification_result": {"species": "African Lion (Panthera leo)"}, "error_message": ""}
    result = species_match(input="x", output=output, expected_output={"species": "African Lion"}, metadata=None)
    assert result.value == "correct"


def test_species_match_wrong():
    output = {"identification_result": {"species": "African Elephant"}, "error_message": ""}
    result = species_match(input="x", output=output, expected_output={"species": "African Lion"}, metadata=None)
    assert result.value == "wrong"


def test_species_match_abstained_is_not_wrong():
    """A confidence-gate abstention must score distinctly from a confident
    misidentification — collapsing the two would hide the difference this
    eval exists to surface."""
    output = {
        "identification_result": {},
        "current_analysis": {"species": "African Lion, but blurry", "confidence_score": 0.2},
        "error_message": ABSTAIN_ERROR,
    }
    result = species_match(input="x", output=output, expected_output={"species": "African Lion"}, metadata=None)
    assert result.value == "abstained"
    assert result.value != "wrong"


def test_species_match_analysis_exception_is_error_not_abstained():
    """node_analyze_image's exception stub (current_analysis == {species:
    'unknown', confidence_score: 0.0}) is disguised by node_unclear_photo_fallback
    as a generic low_confidence error_message — an infra failure must not be
    silently counted as an honest abstention."""
    output = {
        "identification_result": {},
        "current_analysis": {"species": "unknown", "confidence_score": 0.0},
        "error_message": ABSTAIN_ERROR,
    }
    result = species_match(input="x", output=output, expected_output={"species": "African Lion"}, metadata=None)
    assert result.value == "error"


def test_species_match_eval_task_error():
    """run_eval.py's task() catching its own download/graph-invocation
    failure must score as 'error', not silently vanish or count as wrong."""
    output = {
        "identification_result": {}, "current_analysis": {},
        "error_message": f"{EVAL_TASK_ERROR}: connection timed out",
    }
    result = species_match(input="x", output=output, expected_output={"species": "African Lion"}, metadata=None)
    assert result.value == "error"


# ── threat_level_match ────────────────────────────────────────────────────

def test_threat_level_match_correct():
    output = {"identification_result": {"species": "African Lion (Panthera leo)"}, "error_message": ""}
    result = threat_level_match(
        input="x", output=output, expected_output={"species": "African Lion", "threat_level": "high"}, metadata=None,
    )
    assert result.value == "correct"


def test_threat_level_match_abstained():
    output = {"identification_result": {}, "current_analysis": {}, "error_message": ABSTAIN_ERROR}
    result = threat_level_match(
        input="x", output=output, expected_output={"species": "African Lion", "threat_level": "high"}, metadata=None,
    )
    assert result.value == "abstained"


# ── confidence_calibration ────────────────────────────────────────────────

def test_confidence_calibration_reads_raw_current_analysis():
    output = {"current_analysis": {"confidence_score": 0.42}}
    result = confidence_calibration(input="x", output=output, expected_output=None, metadata=None)
    assert result.value == 0.42


def test_confidence_calibration_present_even_on_abstention():
    """current_analysis (the raw per-turn attempt) is populated even when
    identification_result is empty — see state.py."""
    output = {"identification_result": {}, "current_analysis": {"confidence_score": 0.2}, "error_message": ABSTAIN_ERROR}
    result = confidence_calibration(input="x", output=output, expected_output=None, metadata=None)
    assert result.value == 0.2


def test_confidence_calibration_skipped_on_eval_task_error():
    """No genuine stated confidence to calibrate on an eval-harness failure —
    scoring the 0.0 stub would silently pollute the numeric average."""
    output = {"identification_result": {}, "current_analysis": {}, "error_message": f"{EVAL_TASK_ERROR}: boom"}
    assert confidence_calibration(input="x", output=output, expected_output=None, metadata=None) is None


# ── mean_confidence_when_wrong (run evaluator) ────────────────────────────

class _FakeItemResult:
    def __init__(self, evaluations):
        self.evaluations = evaluations


def test_mean_confidence_when_wrong_averages_only_wrong_items():
    item_results = [
        _FakeItemResult([
            Evaluation(name="species_match", value="wrong", data_type="CATEGORICAL"),
            Evaluation(name="confidence_calibration", value=0.8, data_type="NUMERIC"),
        ]),
        _FakeItemResult([
            Evaluation(name="species_match", value="correct", data_type="CATEGORICAL"),
            Evaluation(name="confidence_calibration", value=0.95, data_type="NUMERIC"),
        ]),
        _FakeItemResult([
            Evaluation(name="species_match", value="wrong", data_type="CATEGORICAL"),
            Evaluation(name="confidence_calibration", value=0.6, data_type="NUMERIC"),
        ]),
    ]
    result = mean_confidence_when_wrong(item_results=item_results)
    assert result.value == 0.7  # (0.8 + 0.6) / 2, correct item excluded


def test_mean_confidence_when_wrong_returns_none_when_no_wrong_items():
    item_results = [
        _FakeItemResult([Evaluation(name="species_match", value="correct", data_type="CATEGORICAL")]),
    ]
    assert mean_confidence_when_wrong(item_results=item_results) is None


# ── persona_hallucination ─────────────────────────────────────────────────

_GROUND_TRUTH = {
    "African Lion": {
        "common_name": "African Lion", "scientific_name": "Panthera leo",
        "threat_level": "high", "safety_notes": "Stay in vehicle.",
    }
}


def test_persona_hallucination_skips_abstained_turns():
    """Judging the canned low-confidence retry script against ground truth
    for a species the app never claimed to identify is pure noise."""
    judge_llm = MagicMock()
    evaluator = make_persona_hallucination_evaluator(judge_llm, _GROUND_TRUTH)

    output = {"final_script": "Kate doesn't guess!", "error_message": ABSTAIN_ERROR}
    result = evaluator(input="x", output=output, expected_output={"species": "African Lion"}, metadata=None)

    assert result is None
    judge_llm.with_structured_output.assert_not_called()


def test_persona_hallucination_flags_contradiction():
    judge_llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = MagicMock(contradicts_ground_truth=True, explanation="Said it was harmless.")
    judge_llm.with_structured_output.return_value = structured
    evaluator = make_persona_hallucination_evaluator(judge_llm, _GROUND_TRUTH)

    output = {"final_script": "This gentle giant poses no threat at all.", "error_message": ""}
    result = evaluator(input="x", output=output, expected_output={"species": "African Lion"}, metadata=None)

    assert result.value is True
    assert "harmless" in result.comment


# ── make_task ───────────────────────────────────────────────────────────

def test_task_downloads_image_invokes_graph_and_cleans_up_temp_file():
    graph = MagicMock()
    graph.invoke.return_value = {
        "final_script": "Meet the lion.",
        "identification_result": {"species": "African Lion"},
        "current_analysis": {"species": "African Lion", "confidence_score": 0.9},
        "error_message": "",
    }
    item = MagicMock(input="wildlife/african_lion/x.jpg", id="item-1")

    with patch("wildlens.eval.run_eval._download_image", return_value="C:/tmp/fake.jpg") as mock_dl, \
         patch("wildlens.eval.run_eval.os.unlink") as mock_unlink:
        task = make_task(graph, langfuse_handler=None)
        output = task(item=item)

    mock_dl.assert_called_once_with("wildlife/african_lion/x.jpg")
    mock_unlink.assert_called_once_with("C:/tmp/fake.jpg")
    graph.invoke.assert_called_once()
    assert output["identification_result"]["species"] == "African Lion"


def test_task_returns_error_output_and_cleans_up_temp_file_if_graph_invoke_raises():
    """A task-level exception (transient API error, etc.) must not propagate
    — the langfuse SDK drops any item whose task() raises entirely from
    item_results with no score at all. Returning an error-shaped output
    instead keeps the item in the run, scored distinctly via
    scorers.EVAL_TASK_ERROR."""
    graph = MagicMock()
    graph.invoke.side_effect = RuntimeError("boom")
    item = MagicMock(input="wildlife/african_lion/x.jpg", id="item-1")

    with patch("wildlens.eval.run_eval._download_image", return_value="C:/tmp/fake.jpg"), \
         patch("wildlens.eval.run_eval.os.unlink") as mock_unlink:
        task = make_task(graph, langfuse_handler=None)
        output = task(item=item)

    mock_unlink.assert_called_once_with("C:/tmp/fake.jpg")
    assert output["error_message"].startswith(EVAL_TASK_ERROR)
    assert "boom" in output["error_message"]


def test_task_returns_error_output_when_download_fails():
    graph = MagicMock()
    item = MagicMock(input="wildlife/african_lion/x.jpg", id="item-1")

    with patch("wildlens.eval.run_eval._download_image", side_effect=RuntimeError("network down")), \
         patch("wildlens.eval.run_eval.os.unlink") as mock_unlink:
        task = make_task(graph, langfuse_handler=None)
        output = task(item=item)

    mock_unlink.assert_not_called()  # no temp file was ever created
    graph.invoke.assert_not_called()
    assert output["error_message"].startswith(EVAL_TASK_ERROR)
