# 🧭 DataPilot - AI Data Analysis Agent

Upload a CSV, ask a question in plain English (or Hindi/Hinglish), and the agent writes **pandas** code,
runs it safely, auto-fixes errors, and returns **charts + tables + a short insight**.

```
Question ──► LLM (schema-aware prompt) ──► pandas/plotly code
                                              │
                          AST safety check ◄──┘
                                              │
                        child process (timeout, restricted builtins)
                                              │
                 ok? ──► chart + table + insight      error? ──► send error back to LLM (max 2 retries)
```

## Setup (Windows / Mac / Linux)
```bash
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
The key is read from `.streamlit/secrets.toml` (see below); there is no key box in the UI. **Free option:** Gemini key from https://aistudio.google.com/app/apikey
(provider = "Gemini"). Paid option: Claude key from https://console.anthropic.com (provider = "Claude").

### Set your API key (required, one time)
1. In the project folder (where `app.py` is) create a folder named `.streamlit` (with the dot).
2. Inside it, copy `secrets.toml.example` to `secrets.toml` and put your key:
   `GEMINI_API_KEY = "your-key"`
3. Run `streamlit run app.py` **from the project folder**. Restart the app after editing the file.
`secrets.toml` is git-ignored. Never upload it to GitHub.
Free-tier requests may be used by Google to improve its products, so use only public/sample data.

## Try it
Click **Use sample sales data**, then ask:
- Show monthly sales trend
- Which region has the highest sales? Show a bar chart
- Top 3 products by revenue in 2024
- Average customer rating by product
- Compare sales of Phone vs Laptop month by month

## Files
| File | Job |
|---|---|
| `app.py` | Streamlit UI (upload, chat, charts, downloads) |
| `agent.py` | Prompting, retry loop, insight writing |
| `llm.py` | Gemini (free) and Claude (paid) providers behind one interface |
| `sandbox.py` / `runner.py` | Safety validation + isolated code execution |
| `style.py` | Theme CSS + chart styling |
| `.streamlit/config.toml` | Colour theme, upload limit |
| `data_utils.py` | CSV loading, date detection, schema summary |
| `tests/` | 23 tests (run: `python -m pytest tests`) |

## Security note
Generated code is checked (AST allow-list), run in a separate process with a timeout and restricted builtins,
and API keys are stripped from its environment. For a public deployment, also run in Docker without network.

## Ideas to extend
Excel upload, multiple CSV joins, saving dashboards as PDF, SQL database connector, streaming responses.
