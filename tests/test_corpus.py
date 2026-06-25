import os
from pocketlm import corpus_path, load_corpus

def test_corpus_path_exists():
    path = corpus_path()
    assert os.path.isfile(path)
    assert path.endswith("corpus.txt")

def test_load_corpus_is_nonempty_text():
    text = load_corpus()
    assert isinstance(text, str)
    assert len(text) > 100_000          # the ~1 MB Gutenberg slice
    assert "the" in text.lower()
