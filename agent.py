"""The agent: question -> LLM writes pandas code -> run -> (auto-fix on error) -> answer."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from data_utils import describe_df
from sandbox import ExecResult, run_code

MAX_HISTORY_TURNS = 4

SYSTEM_TEMPLATE = """You are an expert data analyst who answers questions by writing Python (pandas) code.

A pandas DataFrame named `df` is ALREADY loaded in memory. Never read files.

RULES
1. Reply with one or two plain-language sentences describing your approach, then exactly ONE ```python code block.
2. Pre-loaded names: df, pd, np, px (plotly.express), go (plotly.graph_objects). Extra imports allowed ONLY from: math, datetime, statistics, collections, re. No file, network or system access. No matplotlib, no st.*, no input().
3. Do not modify `df` in place; work on df.copy() or new variables.
4. Put the answer table / number / text in a variable named `result` (DataFrame, Series, scalar or str). Aggregate before returning - never return thousands of raw rows.
5. If a chart helps (trend, comparison, distribution, relationship) or the user asks for one, build a plotly figure in a variable named `fig` with a clear title and axis labels. Also set `result` to the aggregated data behind the chart.
6. Use the exact column names from the schema. Convert text to dates with pd.to_datetime when needed. Handle missing values sensibly. Round floats to 2 decimals.
7. Defaults: time series -> line chart sorted by date; rankings -> top 10, sorted descending; use readable labels.
8. If the question cannot be answered from this data, set `result` to a short explanation string.
9. Write explanations and chart titles in the same language as the user's question (English, Hindi or Hinglish).

DATASET SCHEMA
{schema}
"""


@dataclass
class AgentResponse:
    question: str
    explanation: str = ""
    code: str = ""
    raw_reply: str = ""
    execution: ExecResult | None = None
    insight: str = ""
    attempts: int = 0
    model_used: str = ""
    error: str = ""
    log: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.execution and self.execution.ok)


def extract_code(reply: str) -> tuple[str, str]:
    """Return (explanation, code) from an LLM reply."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", reply, re.DOTALL)
    if not m:
        return reply.strip(), ""
    explanation = (reply[: m.start()] + reply[m.end():]).strip()
    return explanation, m.group(1).strip()


def _llm(client, system: str, messages: list[dict], max_tokens: int = 2000) -> str:
    """`client` is any object with .complete(system, messages, max_tokens) -> str (see llm.py)."""
    return client.complete(system, messages, max_tokens)


def _result_preview(ex: ExecResult, max_rows: int = 30) -> str:
    if ex.result_kind == "dataframe" and ex.result_df is not None:
        return ex.result_df.head(max_rows).to_string(index=False)
    return ex.result_text or ""


def _write_insight(client, question: str, ex: ExecResult) -> str:
    preview = _result_preview(ex)
    if not preview:
        return ""
    prompt = (
        f"Question: {question}\n\nResult data:\n{preview}\n\n"
        "Write a 2-4 sentence plain-language insight answering the question with specific numbers. "
        "No headings, no bullet points. Reply in the same language as the question."
    )
    try:
        return _llm(client, "You are a concise data analyst.", [{"role": "user", "content": prompt}], 400).strip()
    except Exception:  # noqa: BLE001 - insight is optional
        return ""


def run_agent(
    client,
    df: pd.DataFrame,
    question: str,
    history: list[dict] | None = None,
    max_retries: int = 2,
    with_insight: bool = True,
    timeout: int = 30,
) -> AgentResponse:
    """history = [{"question": str, "raw_reply": str}, ...] of earlier successful turns."""
    system = SYSTEM_TEMPLATE.format(schema=describe_df(df))
    messages: list[dict] = []
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["raw_reply"]})
    messages.append({"role": "user", "content": question})

    resp = AgentResponse(question=question)
    for attempt in range(1, max_retries + 2):
        resp.attempts = attempt
        reply = _llm(client, system, messages)
        explanation, code = extract_code(reply)
        resp.raw_reply, resp.explanation, resp.code = reply, explanation, code

        if not code:
            resp.error = "The model did not return any code."
            feedback = "You must reply with a ```python code block. Try again."
            ex = ExecResult(ok=False, error=resp.error)
        else:
            ex = run_code(df, code, timeout=timeout)
        resp.execution = ex
        if ex.ok:
            resp.error = ""
            break

        resp.error = ex.error
        resp.log.append(f"Attempt {attempt} failed: {ex.error}")
        feedback = f"Your code failed:\n{ex.error}\n\nFix the problem and reply again with the full corrected code block."
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": feedback}]

    resp.model_used = getattr(client, "used_model", "") or ""
    if resp.ok and with_insight:
        resp.insight = _write_insight(client, question, resp.execution)
    return resp
