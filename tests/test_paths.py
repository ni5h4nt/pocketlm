"""Notebook paths must match cybertect's notebookPath() slug convention so Colab
URLs resolve. This mirrors that rule for the whole course; it asserts only for
unit directories that already exist, so it passes incrementally as units land."""
import re
from pathlib import Path

# Mirror of cybertect src/lib/course-import/slug.ts:
#   idToSlug:     lowercase, drop track./course. prefix, replace "." with "-"
#   slugifyTitle: lowercase, non-alphanumeric runs -> "-", strip leading/trailing "-"

def _id_slug(s: str) -> str:
    return s.lower().split(".", 1)[-1] if s.startswith(("track.", "course.")) else \
        s.lower().replace(".", "-")

def _title_slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

# (unit_id, unit_title, [(lesson_id, lesson_title), ...]) for every unit in the course.
COURSE = [
    ("u0", "Make It Speak", [
        ("l0.1", "Generate text"), ("l0.2", "Tokens"), ("l0.3", "Probabilities"),
        ("l0.4", "Temperature and decoding"), ("l0.5", "Break generation")]),
    ("u1", "Tensors That Learn", [
        ("l1.1", "Tensors and shapes"), ("l1.2", "Matrix multiplication"),
        ("l1.3", "Softmax"), ("l1.4", "Cross-entropy"),
        ("l1.5", "Gradients and autograd"), ("l1.6", "Optimization loop")]),
    ("u2", "Text Becomes Vectors", [
        ("l2.1", "Characters as tokens"), ("l2.2", "Bytes and byte-level BPE"),
        ("l2.3", "Vocabulary design"), ("l2.4", "Embeddings"),
        ("l2.5", "Positional information"), ("l2.6", "Batching and padding")]),
    ("u3", "Attention From Scratch", [
        ("l3.1", "Queries keys and values"), ("l3.2", "Scaled dot-product"),
        ("l3.3", "Causal masking"), ("l3.4", "Multi-head attention"),
        ("l3.5", "Shape tracing and tests")]),
    ("u4", "Assemble the Transformer", [
        ("l4.1", "Residual stream"), ("l4.2", "Pre-normalization"),
        ("l4.3", "MLP feed-forward"), ("l4.4", "Transformer block"),
        ("l4.5", "Output head and weight tying")]),
    ("u5", "Train and Debug It", [
        ("l5.1", "Initialization"), ("l5.2", "Optimizers"),
        ("l5.3", "Learning-rate schedules"), ("l5.4", "Dataset splits"),
        ("l5.5", "Over- and underfitting"), ("l5.6", "Perplexity and evaluation"),
        ("l5.7", "Checkpointing and resume")]),
    ("u6", "Modernize the Architecture", [
        ("l6.1", "RoPE"), ("l6.2", "RMSNorm"), ("l6.3", "SwiGLU"),
        ("l6.4", "GQA/MQA"), ("l6.5", "FlashAttention"), ("l6.6", "Scaling laws"),
        ("l6.7", "MoE introduction"), ("l6.8", "Ablation harness")]),
    ("u7", "Adapt and Post-Train", [
        ("l7.1", "Base vs instruction models"), ("l7.2", "SFT"), ("l7.3", "LoRA"),
        ("l7.4", "QLoRA"), ("l7.5", "Preference optimization"),
        ("l7.6", "Data quality and eval card")]),
    ("u8", "Make Inference Fast", [
        ("l8.1", "KV caching"), ("l8.2", "Memory accounting"),
        ("l8.3", "Quantization"), ("l8.4", "Continuous batching"),
        ("l8.5", "Paged attention"), ("l8.6", "Speculative decoding"),
        ("l8.7", "Serve via inference API")]),
    ("u9", "Model-Engineer Capstone", [
        ("l9.1", "Capstone")]),
]


def _unit_dir(uid: str, utitle: str) -> str:
    return f"{_id_slug(uid)}-{_title_slug(utitle)}"


def _expected(uid: str, utitle: str, lessons) -> set[str]:
    d = _unit_dir(uid, utitle)
    return {f"lessons/{d}/{_id_slug(lid)}-{_title_slug(lt)}.ipynb" for lid, lt in lessons}


def test_existing_unit_notebooks_match_convention():
    checked = 0
    for uid, utitle, lessons in COURSE:
        d = Path("lessons") / _unit_dir(uid, utitle)
        if not d.is_dir():
            continue
        present = {p.as_posix() for p in d.glob("*.ipynb")}
        assert present == _expected(uid, utitle, lessons), f"{uid} paths mismatch"
        checked += 1
    assert checked >= 1   # u0 always exists
