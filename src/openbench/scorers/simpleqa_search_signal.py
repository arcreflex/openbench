import re
from typing import Callable, Any

from inspect_ai.model import get_model, ChatMessageUser, Model
from inspect_ai.scorer import (
    Score,
    Target,
    metric,
    Metric,
    Value,
    SampleScore,
    accuracy,
    stderr,
    scorer,
)
from inspect_ai.solver import TaskState

from .simpleqa import GRADER_TEMPLATE, simpleqa_metrics, simpleqa_scorer


def _extract_tool_calls(state: TaskState) -> tuple[bool, list[str]]:
    """Best-effort detection of assistant tool calls from TaskState.

    Returns (tool_called, tool_names).
    """
    tool_names: list[str] = []

    # Try output.message.tool_calls (preferred if present)
    msg = getattr(state.output, "message", None)
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls or []:
            # tc may be a dict-like or object; try common access patterns
            name = None
            if isinstance(tc, dict):
                name = tc.get("name") or (tc.get("function") or {}).get("name")
            else:
                name = getattr(tc, "name", None)
                if name is None:
                    func = getattr(tc, "function", None)
                    name = getattr(func, "name", None)
            if name:
                tool_names.append(str(name))
        return True, tool_names

    # Fall back to scanning state.messages for assistant messages with tool_calls
    for m in getattr(state, "messages", []) or []:
        if getattr(m, "role", None) == "assistant":
            m_tool_calls = getattr(m, "tool_calls", None)
            if m_tool_calls:
                for tc in m_tool_calls or []:
                    name = None
                    if isinstance(tc, dict):
                        name = tc.get("name") or (tc.get("function") or {}).get("name")
                    else:
                        name = getattr(tc, "name", None)
                        if name is None:
                            func = getattr(tc, "function", None)
                            name = getattr(func, "name", None)
                    if name:
                        tool_names.append(str(name))
                return True, tool_names

    return False, tool_names


@metric
def simpleqa_search_signal_metrics() -> Metric:
    """Calculate tool-aware metrics layered on top of SimpleQA metrics."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        if not scores:
            return {
                "search_call_rate": 0.0,
                "abstain_via_tool_rate": 0.0,
                "attempted_answer_rate": 0.0,
                "hallucination_rate": 0.0,
            }

        total = len(scores)
        tool_called = 0
        abstain_via_tool = 0
        attempted_answers = 0
        incorrect_answers = 0

        for s in scores:
            md = s.score.metadata or {}
            grade = (md.get("grade") or "").lower()
            tool = bool(md.get("tool_called", False))
            abstention_reason = md.get("abstention_reason")

            if tool:
                tool_called += 1
            # Attempted answer means non-abstention (i.e., not NOT_ATTEMPTED)
            if grade in {"correct", "incorrect"}:
                attempted_answers += 1
                if grade == "incorrect":
                    incorrect_answers += 1
            elif tool and abstention_reason == "tool_call":
                abstain_via_tool += 1

        search_call_rate = tool_called / total
        abstain_via_tool_rate = abstain_via_tool / total
        attempted_answer_rate = attempted_answers / total
        hallucination_rate = (
            incorrect_answers / attempted_answers if attempted_answers > 0 else 0.0
        )

        return {
            "search_call_rate": search_call_rate,
            "abstain_via_tool_rate": abstain_via_tool_rate,
            "attempted_answer_rate": attempted_answer_rate,
            "hallucination_rate": hallucination_rate,
        }

    return metric_calculator


@scorer(metrics=[accuracy(), stderr(), simpleqa_metrics(), simpleqa_search_signal_metrics()])
def simpleqa_search_signal_scorer(model: str) -> Callable:
    """SimpleQA scorer variant that treats tool calls as abstention signals.

    If the assistant makes any tool calls and provides no non-empty text output,
    we mark the sample as NOT_ATTEMPTED without invoking the grader. Otherwise,
    we reuse the baseline SimpleQA grader for CORRECT/INCORRECT/NOT_ATTEMPTED.
    """

    # Reuse the baseline grader implementation for non-abstention cases
    baseline_grader = simpleqa_scorer(model)

    async def score(state: TaskState, target: Target) -> Score:
        predicted_answer = (state.output.completion or "").strip()

        # Detect tool calls
        tool_called, tool_names = _extract_tool_calls(state)

        # If a tool was called and there is no answer text, abstain without grading
        if tool_called and not predicted_answer:
            return Score(
                value=0.0,
                answer=predicted_answer,
                metadata={
                    "grade": "not_attempted",
                    "grade_letter": "C",
                    "tool_called": True,
                    "tools": tool_names,
                    "abstention_reason": "tool_call",
                },
            )

        # Otherwise grade using the baseline rubric
        graded = await baseline_grader(state, target)
        # Augment metadata with tool info
        md = dict(graded.metadata or {})
        md.update(
            {
                "tool_called": tool_called,
                "tools": tool_names,
                # If NOT_ATTEMPTED without a tool call, attribute to grader path
                "abstention_reason": (
                    md.get("abstention_reason")
                    or ("grader_not_attempted" if (md.get("grade", "").lower() == "not_attempted" and not tool_called) else None)
                ),
            }
        )

        return Score(value=graded.value, answer=graded.answer, metadata=md)

    return score

