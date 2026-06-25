# PocketLM

Build a decoder-only language model from scratch. PocketLM is a tiny,
pip-installable GPT (char tokenizer + a small transformer + a train loop +
sampling) that trains and generates in seconds on a CPU, plus a set of Colab
notebooks that teach how it works.

You can learn it **standalone** (this repo) **or via cybertect** (the course
that drives these same notebooks). Either way the engine is identical.

## Quickstart

```bash
pip install -e ".[dev]"
```

Then open the first lesson:

```
lessons/u0-make-it-speak/l0-1-generate-text.ipynb
```

Each notebook runs in Colab too. Use the "Open in Colab" path:
`https://colab.research.google.com/github/ni5h4nt/pocketlm/blob/main/lessons/u0-make-it-speak/l0-1-generate-text.ipynb`

## Course index, unit u0 "Make It Speak"

1. `l0-1-generate-text` Generate text
2. `l0-2-tokens` Tokens
3. `l0-3-probabilities` Probabilities
4. `l0-4-temperature-and-decoding` Temperature and decoding
5. `l0-5-break-generation` Break generation

## A note on course structure

The course structure source of truth is the manifest in cybertect; this repo
mirrors it by convention (verified by `tests/test_paths.py`).

## License

MIT, (c) 2026 ni5h4nt.
