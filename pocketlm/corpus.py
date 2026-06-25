"""Access to the bundled training corpus. Shipping the corpus as package data
means a plain `pip install` (including from GitHub in Colab) brings the text with
it, so notebooks never need to clone the repo or change directories."""
from __future__ import annotations
from importlib import resources


def corpus_path() -> str:
    """Absolute path to the bundled corpus file, valid from any directory."""
    return str(resources.files("pocketlm").joinpath("corpus.txt"))


def load_corpus() -> str:
    """Return the bundled corpus as a UTF-8 string."""
    return resources.files("pocketlm").joinpath("corpus.txt").read_text(encoding="utf-8")
