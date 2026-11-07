"""CSV loading + schema description for the LLM."""
from __future__ import annotations

import io
import warnings

import pandas as pd

MAX_UPLOAD_MB = 50


def load_csv(file_or_bytes) -> pd.DataFrame:
    """Read a CSV robustly (encoding fallback, delimiter sniffing, date auto-detection)."""
    raw = file_or_bytes if isinstance(file_or_bytes, (bytes, bytearray)) else file_or_bytes.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_UPLOAD_MB} MB.")

    df, last_err = None, None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if df is None:
        raise ValueError(f"Could not read this file as CSV: {last_err}")

    df.columns = [str(c).strip() for c in df.columns]
    return parse_dates(df)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that look like dates into datetime64."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty or not sample.str.contains(r"[-/:]", regex=True).mean() > 0.9:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.9:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def describe_df(df: pd.DataFrame, max_cols: int = 60) -> str:
    """Compact schema summary that goes into the system prompt."""
    lines = [f"Rows: {len(df):,} | Columns: {df.shape[1]}", "Columns:"]
    for col in list(df.columns)[:max_cols]:
        s = df[col]
        info = f"- `{col}` ({s.dtype}) | nulls: {int(s.isna().sum())}"
        if pd.api.types.is_datetime64_any_dtype(s):
            info += f" | range: {s.min()} to {s.max()}"
        elif pd.api.types.is_bool_dtype(s):
            info += " | boolean"
        elif pd.api.types.is_numeric_dtype(s):
            info += f" | min {s.min():.4g}, mean {s.mean():.4g}, max {s.max():.4g}"
        else:
            n = s.nunique()
            ex = ", ".join(map(str, s.dropna().unique()[:5]))
            info += f" | unique: {n} | examples: {ex}"
        lines.append(info)
    if df.shape[1] > max_cols:
        lines.append(f"... and {df.shape[1] - max_cols} more columns")
    lines.append("\nFirst 5 rows:")
    lines.append(df.head(5).to_string(max_cols=20, max_colwidth=30))
    return "\n".join(lines)


MEASURE_WORDS = ("sales", "revenue", "amount", "profit", "total", "price", "cost", "income", "salary", "score", "rating")


def _is_id_like(df: pd.DataFrame, col: str) -> bool:
    name = str(col).lower()
    if name in ("id", "uid", "uuid", "index") or name.endswith(("_id", " id")) or name.startswith("id_"):
        return True
    s = df[col]
    return pd.api.types.is_integer_dtype(s) and len(df) > 20 and s.nunique() == len(df)


def suggest_questions(df: pd.DataFrame) -> list[str]:
    """Generate a few starter questions from the column types (skips ID columns, prefers money-like measures)."""
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            and not _is_id_like(df, c)]
    def rank(c):  # earlier word in MEASURE_WORDS = better; non-matching columns go last
        hits = [i for i, w in enumerate(MEASURE_WORDS) if w in str(c).lower()]
        return min(hits) if hits else len(MEASURE_WORDS)
    nums.sort(key=rank)  # stable: ties keep original column order
    cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c]) and 1 < df[c].nunique() <= 30]
    q = []
    if dates and nums:
        q.append(f"Show monthly trend of {nums[0]}")
    if cats and nums:
        q.append(f"Top 10 {cats[0]} by total {nums[0]}")
    if len(nums) >= 2:
        q.append(f"Show correlation between {nums[0]} and {nums[1]}")
    if nums:
        q.append(f"Show distribution of {nums[0]}")
    q.append("Give me a quick summary of this dataset")
    return q[:4]
