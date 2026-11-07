"""Safe(ish) execution of LLM-generated pandas code.

Layers of protection:
  1. AST validation   -> rejects dangerous imports / names / attributes before running.
  2. Separate process -> code runs in a child process, so a hang can be killed (timeout).
  3. Restricted builtins inside the child (see runner.py).

NOTE: This is good enough for a portfolio / local project. For a public deployment,
run the whole app inside a Docker container with no network access.
"""
from __future__ import annotations

import ast
import json
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import pandas as pd

ALLOWED_IMPORTS = {
    "pandas", "numpy", "plotly", "math", "datetime", "statistics",
    "collections", "re", "itertools", "functools",
}

BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help",
    "os", "sys", "subprocess", "socket", "shutil", "builtins", "importlib",
    "pathlib", "pickle",
}

# File / network / clipboard I/O methods on pandas & plotly objects
BLOCKED_ATTRS = {
    "read_csv", "read_excel", "read_json", "read_pickle", "read_parquet",
    "read_sql", "read_sql_query", "read_sql_table", "read_html", "read_clipboard",
    "read_table", "read_feather", "read_hdf", "read_fwf", "read_xml", "read_orc",
    "to_csv", "to_excel", "to_json_file", "to_pickle", "to_parquet", "to_sql",
    "to_clipboard", "to_hdf", "to_feather", "to_xml", "to_orc", "to_latex",
    "write_html", "write_image", "write_json", "to_html_file",
    "system", "popen", "spawn", "remove", "unlink", "rmdir", "rmtree",
}

RUNNER = Path(__file__).with_name("runner.py")


class CodeRejected(Exception):
    """Raised when generated code fails the static safety check."""


def _check_module(name: str) -> None:
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise CodeRejected(f"Import of '{name}' is not allowed. Allowed: {sorted(ALLOWED_IMPORTS)}")


def validate_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeRejected(f"SyntaxError on line {e.lineno}: {e.msg}") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise CodeRejected("Relative imports are not allowed.")
            _check_module(node.module or "")
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("__"):
                raise CodeRejected(f"Use of '{node.id}' is not allowed.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in BLOCKED_ATTRS:
                raise CodeRejected(f"Access to attribute '{node.attr}' is not allowed.")


@dataclass
class ExecResult:
    ok: bool
    error: str = ""
    stdout: str = ""
    result_kind: str = "none"          # dataframe | text | none
    result_df: pd.DataFrame | None = None
    result_text: str = ""
    fig_json: str | None = None
    warnings: list[str] = field(default_factory=list)


def run_code(df: pd.DataFrame, code: str, timeout: int = 30) -> ExecResult:
    """Validate and execute `code` against `df` in a child process."""
    try:
        validate_code(code)
    except CodeRejected as e:
        return ExecResult(ok=False, error=f"Rejected by safety check: {e}")

    # Never leak API keys into the child process
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with open(tmp_path / "df.pkl", "wb") as f:
            pickle.dump(df, f)
        (tmp_path / "code.py").write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(RUNNER), str(tmp_path)],
                capture_output=True, text=True, timeout=timeout, env=env,
                cwd=str(RUNNER.parent),
            )
        except subprocess.TimeoutExpired:
            return ExecResult(ok=False, error=f"Code took longer than {timeout}s and was stopped. Use a more efficient approach.")

        out_file = tmp_path / "out.json"
        if not out_file.exists():
            return ExecResult(ok=False, error=f"Runner crashed: {proc.stderr[-1500:] or 'no output'}")
        payload = json.loads(out_file.read_text(encoding="utf-8"))

    res = ExecResult(
        ok=payload["ok"],
        error=payload.get("error", ""),
        stdout=payload.get("stdout", ""),
        result_kind=payload.get("result_kind", "none"),
        result_text=payload.get("result_text", ""),
        fig_json=payload.get("fig_json"),
    )
    if res.result_kind == "dataframe" and payload.get("result_df"):
        res.result_df = pd.read_json(StringIO(payload["result_df"]), orient="split")
    return res
