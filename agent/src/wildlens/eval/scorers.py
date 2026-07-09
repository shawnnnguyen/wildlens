"""
Item-level and run-level Langfuse evaluators for the golden-dataset accuracy eval.

Each item-level evaluator matches langfuse.experiment.EvaluatorFunction's
protocol: (*, input, output, expected_output, metadata, **kwargs) -> Evaluation
| list[Evaluation] | None. `output` is exactly what run_eval.py's task()
returns for that item — see its docstring for the shape:
    {"final_script": str, "identification_result": dict,
     "current_analysis": dict, "error_message": str}

`expected_output` is the dataset item's {"species": ..., "threat_level": ...}
set by seed_dataset.py from species_list.json.
"""
from __future__ import annotations

from typing import Any

from langfuse import Evaluation

from ..data.species_lookup import canonical_common_name, ground_truth_threat_level

# node_unclear_photo_fallback's sentinel (nodes.py) for a confidence-gate
# abstention — see graphs.py's route_after_analysis / MIN_CONFIDENCE.
ABSTAIN_ERROR = "low_confidence"

# run_eval.py's task() prefixes error_message with this when the eval harness
# itself failed (image download, graph invocation raised) — an infra problem,
# not a model judgment. See EVAL_TASK_ERROR usage in run_eval.py.
EVAL_TASK_ERROR = "eval_task_error"


def _is_eval_task_error(output: dict) -> bool:
    return output.get("error_message", "").startswith(EVAL_TASK_ERROR)


def _is_analysis_exception(output: dict) -> bool:
    """
    node_analyze_image (nodes.py) sets current_analysis to the exact stub
    {"confidence_score": 0.0, "species": "unknown"} on ANY exception (Gemini
    API error, malformed structured output, network failure) — see its
    except branch. node_unclear_photo_fallback then unconditionally
    overwrites error_message to the generic ABSTAIN_ERROR sentinel
    regardless of cause (nodes.py), which loses the original exception
    message. This stub shape is the only surviving signal that lets this
    eval tell "Gemini genuinely returned a low-confidence read" apart from
    "node_analyze_image blew up" — both would otherwise score identically
    as an honest abstention, which they are not: an infra failure on a
    golden-set item should never be silently counted as the model
    correctly declining to guess.

    Heuristic, not exact: relies on nodes.py's current exception-stub
    contract rather than a dedicated signal threaded through state. Gemini
    is prompted to always return an actual species guess (just with lower
    confidence) for a blurry photo, so a literal "unknown" is not expected
    from a genuine model response — but if nodes.py's stub shape ever
    changes, this heuristic needs to change with it.
    """
    if output.get("error_message") != ABSTAIN_ERROR:
        return False
    analysis = output.get("current_analysis", {})
    return analysis.get("species") == "unknown" and analysis.get("confidence_score") == 0.0


def _is_abstention(output: dict) -> bool:
    return output.get("error_message") == ABSTAIN_ERROR and not _is_analysis_exception(output)


def _error_evaluation(name: str, output: dict) -> Evaluation | None:
    """
    Shared 'error' branch for species_match/threat_level_match: an eval-harness
    infra failure (run_eval.py's task() couldn't download the image or invoke
    the graph) or a node_analyze_image exception disguised by the fallback
    node as a generic low-confidence abstention (see _is_analysis_exception).
    Scored as 'error' — distinct from both 'abstained' (an honest low-confidence
    decline) and 'wrong' (a confident misidentification) — so a flaky run
    doesn't inflate either of those counts. Returns None when neither applies,
    so callers can `if (e := _error_evaluation(...)) is not None: return e`.
    """
    if _is_eval_task_error(output):
        return Evaluation(
            name=name, value="error", data_type="CATEGORICAL",
            comment=f"Eval harness failure, not a model judgment: {output.get('error_message')}",
        )
    if _is_analysis_exception(output):
        return Evaluation(
            name=name, value="error", data_type="CATEGORICAL",
            comment="node_analyze_image raised an exception — not a genuine low-confidence read.",
        )
    return None


def species_match(*, input: Any, output: dict, expected_output: Any, metadata: Any, **_: Any) -> Evaluation:
    """
    'abstained' is scored distinctly from 'wrong': an app that declines to
    guess on a hard photo (MIN_CONFIDENCE gate fired) is not the same failure
    mode as one that confidently misidentifies the animal, and this product's
    entire value proposition is correct identification — collapsing the two
    into one score would hide the difference this eval exists to surface.
    """
    expected_species = (expected_output or {}).get("species")

    error_eval = _error_evaluation("species_match", output)
    if error_eval is not None:
        return error_eval

    if _is_abstention(output):
        raw_guess = output.get("current_analysis", {}).get("species")
        return Evaluation(
            name="species_match",
            value="abstained",
            data_type="CATEGORICAL",
            comment=f"Confidence gate fired (raw guess: {raw_guess!r})",
        )

    actual_raw = output.get("identification_result", {}).get("species", "")
    actual = canonical_common_name(actual_raw)
    is_match = actual is not None and actual == expected_species
    return Evaluation(
        name="species_match",
        value="correct" if is_match else "wrong",
        data_type="CATEGORICAL",
        comment=f"expected={expected_species!r} actual={(actual or actual_raw)!r}",
    )


def threat_level_match(*, input: Any, output: dict, expected_output: Any, metadata: Any, **_: Any) -> Evaluation:
    """Uses the curated ground_truth_threat_level() lookup (not Gemini's live
    call) for 'actual', consistent with node_analyze_image's own escalation
    check — this scores whether the curated data for whatever species Gemini
    landed on matches the expected species' curated threat_level."""
    expected_level = (expected_output or {}).get("threat_level")

    error_eval = _error_evaluation("threat_level_match", output)
    if error_eval is not None:
        return error_eval

    if _is_abstention(output):
        return Evaluation(name="threat_level_match", value="abstained", data_type="CATEGORICAL")

    actual_raw = output.get("identification_result", {}).get("species", "")
    actual_level = ground_truth_threat_level(actual_raw)
    is_match = actual_level is not None and actual_level == expected_level
    return Evaluation(
        name="threat_level_match",
        value="correct" if is_match else "wrong",
        data_type="CATEGORICAL",
        comment=f"expected={expected_level!r} actual={actual_level!r}",
    )


def confidence_calibration(*, input: Any, output: dict, expected_output: Any, metadata: Any, **_: Any) -> Evaluation | None:
    """Raw confidence_score for this item regardless of outcome (from
    current_analysis, the raw per-turn attempt — see state.py — so this is
    populated even on an abstention). Paired with the mean_confidence_when_wrong
    run-evaluator below. Skipped (None) on an eval-harness/analysis error —
    there's no genuine stated confidence to calibrate against in that case,
    just a 0.0 stub that would otherwise silently pollute the numeric average."""
    if _is_eval_task_error(output) or _is_analysis_exception(output):
        return None
    confidence = output.get("current_analysis", {}).get("confidence_score", 0.0)
    return Evaluation(name="confidence_calibration", value=float(confidence), data_type="NUMERIC")


def mean_confidence_when_wrong(*, item_results: list, **_: Any) -> Evaluation | None:
    """
    Run-level aggregate: mean stated confidence across items scored 'wrong'
    (not 'abstained') by species_match. A well-calibrated model should show
    LOW confidence on its wrong answers; a high mean here means the model is
    confidently wrong, which is worse than an honest abstention.

    Returns None (no score recorded) when there are no wrong items in this
    run — the langfuse SDK normalizes a None return to "no evaluation", not
    an error (see langfuse.experiment._run_evaluator).
    """
    wrong_confidences = []
    for result in item_results:
        species_eval = next((e for e in result.evaluations if e.name == "species_match"), None)
        confidence_eval = next((e for e in result.evaluations if e.name == "confidence_calibration"), None)
        if species_eval is not None and species_eval.value == "wrong" and confidence_eval is not None:
            wrong_confidences.append(confidence_eval.value)

    if not wrong_confidences:
        return None
    return Evaluation(
        name="mean_confidence_when_wrong",
        value=sum(wrong_confidences) / len(wrong_confidences),
        data_type="NUMERIC",
        comment=f"n={len(wrong_confidences)}",
    )


def make_persona_hallucination_evaluator(judge_llm, species_ground_truth: dict[str, dict]):
    """
    Factory returning an LLM-as-judge evaluator bound to *judge_llm* — a
    dependency-injection-via-closure pattern matching how nodes.py binds
    llm/retriever into graph nodes (see graphs.py's build_graph), so this
    stays independently testable with a mocked judge.

    judge_llm should be a different provider than the persona LLM (DeepSeek)
    to avoid self-grading — run_eval.py wires this to Gemini, already a
    dependency for node_analyze_image.

    species_ground_truth: {common_name: species_list.json entry}, used to
    give the judge the safety_notes/threat_level to check the script against.
    """
    from pydantic import BaseModel, Field

    class _Judgment(BaseModel):
        contradicts_ground_truth: bool = Field(
            description="True if the script states something that contradicts the "
                        "provided ground truth (wrong threat level, invented safety "
                        "claims, factually wrong habitat/behavior)."
        )
        explanation: str = Field(description="One sentence: what, if anything, was contradicted.")

    def persona_hallucination(*, input: Any, output: dict, expected_output: Any, metadata: Any, **_: Any) -> Evaluation | None:
        # Judging the canned "Baako doesn't guess!" retry script (see
        # node_unclear_photo_fallback) against ground truth for an animal the
        # app never actually claimed to identify is pure noise — skip it.
        if _is_abstention(output):
            return None

        expected_species = (expected_output or {}).get("species")
        entry = species_ground_truth.get(expected_species)
        script = output.get("final_script", "")
        if not entry or not script:
            return None

        ground_truth_text = (
            f"Species: {entry['common_name']} ({entry.get('scientific_name', '')})\n"
            f"Threat level: {entry['threat_level']}\n"
            f"Safety notes: {entry.get('safety_notes', '')}"
        )
        prompt = (
            "Compare the following safari-guide narration script against the ground "
            "truth facts for the species it is supposedly describing. Flag it only if "
            "it states something that actively CONTRADICTS the ground truth (not "
            "merely omits detail).\n\n"
            f"GROUND TRUTH:\n{ground_truth_text}\n\nSCRIPT:\n{script}"
        )
        judgment = judge_llm.with_structured_output(_Judgment).invoke(prompt)
        return Evaluation(
            name="persona_hallucination",
            value=bool(judgment.contradicts_ground_truth),
            data_type="BOOLEAN",
            comment=judgment.explanation,
        )

    return persona_hallucination
