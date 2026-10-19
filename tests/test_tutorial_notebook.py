"""Smoke test: the tutorial notebook executes end to end (in a temporary copy)."""
import shutil
from pathlib import Path

import nbformat
from nbclient import NotebookClient

SRC = Path(__file__).resolve().parents[1] / "tutorial_notebook"


def test_tutorial_notebook_runs(tmp_path):
    work = tmp_path / "tutorial_notebook"
    shutil.copytree(SRC, work, ignore=shutil.ignore_patterns("__pycache__"))
    nb = nbformat.read(work / "signed_inference_tutorial.ipynb", as_version=4)
    NotebookClient(
        nb, timeout=900, kernel_name="python3",
        resources={"metadata": {"path": str(work)}},
    ).execute()  # raises if any cell fails
