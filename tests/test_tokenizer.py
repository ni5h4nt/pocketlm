from pocketlm.tokenizer import CharTokenizer

def test_roundtrip_in_vocab():
    t = CharTokenizer.from_text("hello world")
    assert t.decode(t.encode("hello world")) == "hello world"

def test_oov_maps_to_unk():
    t = CharTokenizer.from_text("abc")
    ids = t.encode("aZc")           # Z is OOV
    assert ids[1] == t.unk_id
    assert t.decode(ids) == "a�c"

def test_unicode_is_per_codepoint():
    t = CharTokenizer.from_text("ab")
    # emoji is OOV here; one codepoint -> one token -> one <unk>
    assert t.encode("a\U0001f600b") == [t.stoi["a"], t.unk_id, t.stoi["b"]]

def test_vocab_size_includes_unk():
    t = CharTokenizer.from_text("abc")
    assert t.vocab_size == 4        # a,b,c + <unk>

def test_sentinel_in_corpus_does_not_collide():
    t = CharTokenizer.from_text("ab�c")     # corpus literally contains the sentinel
    assert t.itos.count("�") == 1           # only the reserved unk slot
    assert t.encode("�")[0] == t.unk_id     # treated as OOV
