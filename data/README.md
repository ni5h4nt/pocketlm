# Corpus provenance

- **Source:** Project Gutenberg eBook #100 (The Complete Works of William
  Shakespeare), public domain.
- **URL:** https://www.gutenberg.org/files/100/100-0.txt
- **Retrieved:** 2026-06-24
- **Location:** `pocketlm/corpus.txt` (shipped as package data; load it with
  `pocketlm.load_corpus()` or `pocketlm.corpus_path()`).
- **SHA-256 of `corpus.txt`:** `01bfff14d1dcda529b53fb34f0b0c761d1a66a46b45b557d1837804302b75b14`
- **Size:** 1,012,748 bytes UTF-8 (vocab 86 distinct characters)
- **Transformation:** strip the Gutenberg START/END boilerplate, take the leading
  1,000,000 characters.

Rebuild with `python scripts/fetch_corpus.py` (run from the repo root).

**Swappable:** replace with any public-domain Gutenberg work and update this file.

Note: the SHA-256 is valid for the file as fetched on the recorded date. If
Gutenberg revises eBook #100 the hash may change; the committed `corpus.txt` is
the canonical artifact, the script just rebuilds it.
