"""DataPilot - chat with your CSV.   Run:  streamlit run app.py"""
import inspect
import os
from pathlib import Path

import plotly.io as pio
import streamlit as st

from agent import AgentResponse, run_agent
from data_utils import describe_df, load_csv, suggest_questions
from llm import CLAUDE_DEFAULT, GEMINI_MODELS, LLMError, make_client
from style import CSS, esc, style_fig

st.set_page_config(page_title="DataPilot - chat with your CSV", page_icon="🧭", layout="wide")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

SAMPLE = Path(__file__).with_name("sample_sales.csv")
# newer Streamlit uses width="stretch", older uses use_container_width=True
STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}


def get_secret(name: str) -> str:
    """Key from .streamlit/secrets.toml (or Streamlit Cloud secrets), else environment variable."""
    try:
        value = st.secrets.get(name)
    except Exception:  # noqa: BLE001 - no secrets file present
        value = None
    return str(value or os.getenv(name, "") or "")


def hero(compact: bool = False):
    st.markdown(
        f"""<div class="hero {'compact' if compact else ''}">
        <h1>Ask your data. <span>Get answers, not code.</span></h1>
        <p>Upload a CSV and ask in plain English, Hindi or Hinglish. DataPilot writes the pandas code,
        runs it in a sandbox and brings back a chart, a table and a plain-language insight.</p></div>""",
        unsafe_allow_html=True)


def render_response(r: AgentResponse, idx: int):
    if r.explanation:
        st.caption(r.explanation)
    if not r.ok:
        st.error(f"Could not complete this analysis after {r.attempts} attempt(s).\n\n{r.error}")
        if r.code:
            with st.expander("Last code that failed"):
                st.code(r.code, language="python")
        return

    ex = r.execution
    tags = '<span class="tag ok">✓ verified in sandbox</span>'
    tags += (f'<span class="tag warn">⟲ auto-fixed in {r.attempts} attempts</span>' if r.attempts > 1
             else '<span class="tag">⚡ solved first try</span>')
    if r.model_used:
        tags += f'<span class="tag">⚙ {esc(r.model_used)}</span>'
    st.markdown(f'<div class="tags">{tags}</div>', unsafe_allow_html=True)

    if r.insight:
        st.markdown(f'<div class="insight"><b>INSIGHT</b>{esc(r.insight)}</div>', unsafe_allow_html=True)

    if ex.result_kind == "text" and ex.result_text:
        st.markdown(f'<div class="bigstat">{esc(ex.result_text)}</div>' if len(ex.result_text) < 40
                    else f"<div>{esc(ex.result_text)}</div>", unsafe_allow_html=True)

    tab_names, has_fig = [], bool(ex.fig_json)
    if has_fig:
        tab_names.append("📈 Chart")
    if ex.result_df is not None:
        tab_names.append("🧾 Table")
    tab_names.append("🧠 Code")
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    if has_fig:
        with tabs["📈 Chart"]:
            st.plotly_chart(style_fig(pio.from_json(ex.fig_json)), key=f"fig_{idx}")
    if ex.result_df is not None:
        with tabs["🧾 Table"]:
            st.dataframe(ex.result_df, key=f"df_{idx}")
            st.download_button("⬇ Download CSV", ex.result_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"result_{idx + 1}.csv", mime="text/csv", key=f"dl_{idx}")
    with tabs["🧠 Code"]:
        st.code(r.code, language="python")
        st.download_button("⬇ Download code", r.code.encode("utf-8"), file_name=f"analysis_{idx + 1}.py",
                           mime="text/x-python", key=f"code_{idx}")
        if ex.stdout.strip():
            st.caption("Printed output")
            st.code(ex.stdout)


# ---------------- AI provider (configured in secrets, not shown in the UI) ----------------
if get_secret("GEMINI_API_KEY"):
    provider, api_key, model = "gemini", get_secret("GEMINI_API_KEY"), get_secret("GEMINI_MODEL") or GEMINI_MODELS[0]
elif get_secret("ANTHROPIC_API_KEY"):
    provider, api_key, model = "claude", get_secret("ANTHROPIC_API_KEY"), get_secret("ANALYST_MODEL") or CLAUDE_DEFAULT
else:
    provider, api_key, model = "gemini", "", GEMINI_MODELS[0]
MAX_QUESTIONS = int(get_secret("MAX_QUESTIONS") or 25)  # protects your free quota on a public link

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown('<div class="brand">🧭 DataPilot<small>YOUR CSV CO-PILOT</small></div>', unsafe_allow_html=True)

    if not api_key:
        st.warning("API key not found. Add GEMINI_API_KEY to .streamlit/secrets.toml and restart the app.")

    st.markdown('<div class="label">Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a CSV", type=["csv"], label_visibility="collapsed")
    if SAMPLE.exists() and st.button("✨ Try sample sales data", **STRETCH):
        st.session_state.df = load_csv(SAMPLE.read_bytes())
        st.session_state.filename = "sample_sales.csv"
        st.session_state.history = []

    st.markdown('<div class="label">Options</div>', unsafe_allow_html=True)
    with_insight = st.toggle("Write an insight with each answer", value=True)
    if st.button("🗑 Clear conversation", **STRETCH):
        st.session_state.history = []

if uploaded is not None and st.session_state.get("filename") != uploaded.name:
    try:
        st.session_state.df = load_csv(uploaded)
        st.session_state.filename = uploaded.name
        st.session_state.history = []
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(str(e))

st.session_state.setdefault("history", [])
df = st.session_state.get("df")

# ---------------- empty state ----------------
if df is None:
    hero()
    st.markdown(
        """<div class="pipe">
        <span class="node">💬 Your question</span><span class="arrow">→</span>
        <span class="node">🤖 AI writes pandas</span><span class="arrow">→</span>
        <span class="node">🛡 Safety check</span><span class="arrow">→</span>
        <span class="node">⚙ Sandbox run</span><span class="arrow">→</span>
        <span class="node">📈 Chart + insight</span></div>
        <div class="grid">
          <div class="fcard"><div class="n">01</div><h3>Upload</h3>
            <p>Drop any CSV. Dates, delimiters and encodings are detected automatically.</p></div>
          <div class="fcard"><div class="n">02</div><h3>Ask</h3>
            <p>“Show monthly sales trend”, “top 10 products”, or in Hinglish: “region wise sales dikhao”.</p></div>
          <div class="fcard"><div class="n">03</div><h3>Understand</h3>
            <p>Get an interactive chart, the result table and the exact code, so nothing is a black box.</p></div>
        </div>
        <div class="hint">👈 No CSV handy? Press <b>Try sample sales data</b> in the sidebar to start.</div>""",
        unsafe_allow_html=True)
    st.stop()

# ---------------- data loaded ----------------
hero(compact=True)
num_cols = sum(1 for c in df.columns if str(df[c].dtype).startswith(("int", "float", "Int", "Float", "uint")))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", df.shape[1])
c3.metric("Numeric columns", num_cols)
c4.metric("Missing values", f"{df.isna().mean().mean() * 100:.1f}%")
st.caption(f"Dataset: {st.session_state.filename}")

with st.expander("🔎 Preview data & column info"):
    st.dataframe(df.head(50))
    st.text(describe_df(df).split("\n\nFirst 5 rows:")[0])

st.markdown('<div class="label">Try asking</div>', unsafe_allow_html=True)
pending = None
cols = st.columns(4)
for i, q in enumerate(suggest_questions(df)):
    if cols[i % 4].button(q, key=f"sugg_{i}", **STRETCH):
        pending = q

for i, turn in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant", avatar="🧭"):
        render_response(turn["response"], i)

question = st.chat_input("Ask about your data, e.g. “show monthly sales trend”") or pending
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant", avatar="🧭"):
        if not api_key:
            st.error("No API key configured. Add GEMINI_API_KEY to .streamlit/secrets.toml and restart.")
        elif len(st.session_state.history) >= MAX_QUESTIONS:
            st.warning(f"Question limit reached for this session ({MAX_QUESTIONS}). Use 'Clear conversation' to continue.")
        else:
            try:
                with st.spinner("Thinking → writing code → running it safely…"):
                    prior = [{"question": t["question"], "raw_reply": t["response"].raw_reply}
                             for t in st.session_state.history if t["response"].ok]
                    resp = run_agent(make_client(provider, api_key, model), df, question,
                                     history=prior, with_insight=with_insight)
                idx = len(st.session_state.history)
                st.session_state.history.append({"question": question, "response": resp})
                render_response(resp, idx)
            except LLMError as e:
                st.error(str(e))
            except Exception as e:  # noqa: BLE001
                st.error(f"Unexpected error: {e}")
