"""Child-process entry point. Runs generated code with restricted builtins.

Usage: python runner.py <workdir>   (workdir contains df.pkl + code.py, receives out.json)
"""
import builtins
import contextlib
import io
import json
import pickle
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sandbox import ALLOWED_IMPORTS, validate_code

MAX_ROWS = 1000

SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "bin", "callable", "chr", "dict", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "hex", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "oct", "ord", "pow", "print",
    "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "zip", "Exception", "ValueError", "KeyError", "TypeError", "IndexError",
    "ZeroDivisionError", "StopIteration", "ArithmeticError", "AttributeError",
]


def make_builtins():
    real_import = builtins.__import__

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 or name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(f"Import of '{name}' is not allowed")
        return real_import(name, globals, locals, fromlist, level)

    safe = {n: getattr(builtins, n) for n in SAFE_BUILTIN_NAMES}
    safe["__import__"] = restricted_import
    return safe


def serialize_result(result):
    if result is None:
        return {"result_kind": "none"}
    if isinstance(result, pd.Series):
        result = result.rename(result.name if result.name is not None else "value").reset_index()
    if isinstance(result, pd.DataFrame):
        df = result.head(MAX_ROWS)
        try:
            return {"result_kind": "dataframe", "result_df": df.to_json(orient="split", date_format="iso")}
        except Exception:
            return {"result_kind": "dataframe", "result_df": df.astype(str).to_json(orient="split")}
    if hasattr(result, "item"):
        try:
            result = result.item()
        except Exception:
            pass
    return {"result_kind": "text", "result_text": str(result)}


def error_message(exc: BaseException, code: str) -> str:
    lineno = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<analysis>":
            lineno = frame.lineno
    msg = f"{type(exc).__name__}: {exc}"
    if lineno:
        lines = code.splitlines()
        if 0 < lineno <= len(lines):
            msg += f"\n(on line {lineno}: {lines[lineno - 1].strip()})"
    return msg


def main(workdir: Path):
    code = (workdir / "code.py").read_text(encoding="utf-8")
    with open(workdir / "df.pkl", "rb") as f:
        df = pickle.load(f)

    out = {"ok": False, "stdout": ""}
    try:
        validate_code(code)
        env = {"__builtins__": make_builtins(), "pd": pd, "np": np, "px": px, "go": go, "df": df.copy()}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(code, "<analysis>", "exec"), env)
        out["stdout"] = buf.getvalue()[:5000]
        out.update(serialize_result(env.get("result")))

        fig = env.get("fig")
        if fig is not None:
            if isinstance(fig, go.Figure):
                out["fig_json"] = fig.to_json()
            else:
                raise TypeError("`fig` must be a plotly Figure (use px.* or go.Figure), not another type")
        if out.get("result_kind", "none") == "none" and out.get("fig_json") is None:
            raise ValueError("Code produced no output. Assign the answer to `result` and/or a plotly figure to `fig`.")
        out["ok"] = True
    except BaseException as e:  # noqa: BLE001 - we want to report everything back to the LLM
        out["ok"] = False
        out["error"] = error_message(e, code)

    (workdir / "out.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
