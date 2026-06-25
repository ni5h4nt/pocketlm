"""Fetch Gutenberg eBook #100, strip license boilerplate, write a ~1 MB slice."""
import hashlib
import urllib.request

URL = "https://www.gutenberg.org/files/100/100-0.txt"
START = "*** START OF THE PROJECT GUTENBERG EBOOK"
END = "*** END OF THE PROJECT GUTENBERG EBOOK"
SLICE_CHARS = 1_000_000   # ~1 MB for mostly-ASCII text (str slicing is by codepoint)

def build() -> str:
    raw = urllib.request.urlopen(URL).read().decode("utf-8")
    after_start = raw.split(START, 1)[1]            # everything past the START marker
    body = after_start[after_start.index("\n") + 1:]  # drop the rest of the START line
    body = body.split(END, 1)[0].strip()
    return body[:SLICE_CHARS]

if __name__ == "__main__":
    text = build()
    with open("data/corpus.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("bytes:", len(text.encode("utf-8")))
    print("sha256:", hashlib.sha256(text.encode("utf-8")).hexdigest())
    print("vocab:", len(set(text)))
