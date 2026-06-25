import re
from pathlib import Path

def _slug(s: str) -> str:
    return re.sub(r"-+$", "", re.sub(r"[^a-z0-9]+", "-", s.lower()))

U0 = [("l0.1", "Generate text"), ("l0.2", "Tokens"), ("l0.3", "Probabilities"),
      ("l0.4", "Temperature and decoding"), ("l0.5", "Break generation")]

def _expected() -> set[str]:
    unit = "u0-make-it-speak"
    return {f"lessons/{unit}/{lid.replace('.', '-')}-{_slug(title)}.ipynb"
            for lid, title in U0}

def test_u0_notebook_paths_match_convention():
    present = {p.as_posix() for p in Path("lessons/u0-make-it-speak").glob("*.ipynb")}
    assert present == _expected()
