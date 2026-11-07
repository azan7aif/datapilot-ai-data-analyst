import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from agent import extract_code, run_agent
from data_utils import load_csv, suggest_questions
from sandbox import CodeRejected, run_code, validate_code

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df():
    return load_csv((ROOT / "sample_sales.csv").read_bytes())


class FakeClient:
    """Mimics anthropic.Anthropic().messages.create with scripted replies."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.last_messages = None

    def complete(self, system, messages, max_tokens=2000):
        self.calls += 1
        self.last_messages = messages
        return self.replies.pop(0) if self.replies else "Done."


TREND = """I'll group sales by month.
```python
d = df.copy()
d["month"] = d["order_date"].dt.to_period("M").dt.to_timestamp()
result = d.groupby("month", as_index=False)["sales"].sum().round(2)
fig = px.line(result, x="month", y="sales", title="Monthly sales")
```"""


def test_csv_dates_detected(df):
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
    qs = suggest_questions(df)
    assert len(qs) >= 3
    assert not any("order_id" in q for q in qs)        # ID columns are not useful to analyse
    assert "sales" in qs[0]                            # money-like column is preferred


def test_extract_code():
    expl, code = extract_code(TREND)
    assert "group sales" in expl and "px.line" in code


def test_trend_runs_and_returns_chart(df):
    ex = run_code(df, extract_code(TREND)[1])
    assert ex.ok, ex.error
    assert ex.fig_json and ex.result_df is not None and len(ex.result_df) >= 20


def test_scalar_result(df):
    ex = run_code(df, "result = df['sales'].sum()")
    assert ex.ok and ex.result_kind == "text"


@pytest.mark.parametrize("bad", [
    "import os\nresult = os.listdir('.')",
    "result = open('/etc/passwd').read()",
    "result = eval('1+1')",
    "df.to_csv('x.csv')\nresult = 1",
    "result = ().__class__.__bases__",
    "import subprocess\nresult = 1",
    "result = pd.read_csv('sample_sales.csv')",
    "result = getattr(df, 'to_csv')",
])
def test_dangerous_code_rejected(bad, df):
    with pytest.raises(CodeRejected):
        validate_code(bad)
    assert not run_code(df, bad).ok


def test_timeout(df):
    ex = run_code(df, "while True:\n    pass\nresult = 1", timeout=3)
    assert not ex.ok and "longer than" in ex.error


def test_no_output_is_error(df):
    assert not run_code(df, "x = 1").ok


def test_runtime_error_has_line_info(df):
    ex = run_code(df, "result = df['nope'].sum()")
    assert not ex.ok and "KeyError" in ex.error and "line 1" in ex.error


def test_agent_self_heals(df):
    broken = "Trying.\n```python\nresult = df['revenue'].sum()\n```"
    client = FakeClient([broken, TREND, "Sales rose steadily."])
    r = run_agent(client, df, "show monthly sales trend")
    assert r.ok and r.attempts == 2 and r.execution.fig_json
    assert r.insight == "Sales rose steadily."


def test_agent_gives_up(df):
    broken = "x\n```python\nresult = df['revenue'].sum()\n```"
    r = run_agent(FakeClient([broken] * 5), df, "q", max_retries=1)
    assert not r.ok and r.attempts == 2 and "KeyError" in r.error


def test_followup_history_is_sent(df):
    client = FakeClient([TREND])
    run_agent(client, df, "now by region", history=[{"question": "trend", "raw_reply": TREND}], with_insight=False)
    assert [m["role"] for m in client.last_messages] == ["user", "assistant", "user"]
