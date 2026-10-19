"""Smoke test: the tutorial notebook executes end to end."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK = (Path(__file__).resolve().parents[1]
            / "tutorial_notebook" / "signed_inference_tutorial.ipynb")


def test_tutorial_notebook_runs():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        nb, timeout=900, kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    client.execute()  # raises if any cell fails
