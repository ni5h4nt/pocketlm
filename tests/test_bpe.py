from pocketlm.bpe import BPETokenizer

# Two pangrams: enough distinct adjacent pairs to support many merges.
RICH = ("the quick brown fox jumps over the lazy dog. "
        "pack my box with five dozen liquor jugs. ") * 100

def test_roundtrip_ascii():
    t = BPETokenizer().train(RICH, 300)
    s = "the quick brown fox"
    assert t.decode(t.encode(s)) == s

def test_roundtrip_unicode():
    t = BPETokenizer().train("héllo wörld " * 30, 300)
    s = "héllo \U0001f600 wörld"   # accents + emoji survive byte-level
    assert t.decode(t.encode(s)) == s

def test_vocab_size_reaches_target():
    t = BPETokenizer().train(RICH, 300)
    assert t.vocab_size == 300

def test_merges_shorten_repetitive_text():
    text = "abcabcabc" * 100
    t = BPETokenizer().train(text, 300)
    assert len(t.encode(text)) < len(text.encode("utf-8"))

def test_untrained_is_raw_bytes():
    t = BPETokenizer()
    assert t.encode("hi") == list("hi".encode("utf-8"))
    assert t.vocab_size == 256

def test_vocab_size_floor():
    import pytest
    with pytest.raises(ValueError):
        BPETokenizer().train("x", 100)   # below the byte alphabet
