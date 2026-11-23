"""Three retrievers over the P1 corpus: BM25, dense (bge-m3 via Ollama) and RRF.

Everything is small on purpose, so each part can be read and explained:

    load_corpus()   documents from data/corpus, as {path: text}
    chunk()         splits a document by headings, then into ~300-word windows
    tokenize()      lowercase words, keeping identifiers like immich_postgres whole
    BM25            the classic lexical ranking, written out (~30 lines)
    DenseIndex      cosine similarity over embeddings, cached on disk
    rrf()           reciprocal rank fusion of several document rankings

Retrievers rank chunks; `to_doc_ranking()` turns that into a ranking of
documents (a document scores as its best chunk), because the golden set
labels documents, not chunks. RRF fuses those document rankings.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_CORPUS = REPO_ROOT / "data" / "corpus"
DEFAULT_CACHE = REPO_ROOT / "data" / "cache"

CHUNK_WORDS = 300
CHUNK_OVERLAP = 50
RRF_K = 60

# ---------------------------------------------------------------- corpus


@dataclass(frozen=True)
class Chunk:
    doc: str  # corpus path, same form as `relevant_docs` in the golden set
    n: int  # position of the chunk inside its document
    text: str

    @property
    def key(self) -> str:
        """Content hash: the cache key for its embedding."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def load_corpus(corpus: Path = DEFAULT_CORPUS) -> dict[str, str]:
    manifest = corpus / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{manifest} not found: run projects/p1-rag-evals-lab/fetch_corpus.py first"
        )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        f["path"]: (corpus / f["path"]).read_text(encoding="utf-8", errors="replace")
        for s in data["sources"]
        for f in s["files"]
    }


HEADING = re.compile(r"^#{1,6} ", re.MULTILINE)


def chunk(
    doc: str, text: str, words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Split at Markdown headings, then cut long sections into overlapping windows.

    Headings keep a chunk about one topic. The window keeps chunks short enough
    for the embedding to stay specific; the overlap stops an answer that sits
    on a boundary from being cut in half.
    """
    starts = [m.start() for m in HEADING.finditer(text)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    sections = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)], strict=True)]

    out: list[Chunk] = []
    step = words - overlap
    for section in sections:
        tokens = section.split()
        if not tokens:
            continue
        for i in range(0, max(len(tokens) - overlap, 1), step):
            out.append(Chunk(doc, len(out), " ".join(tokens[i : i + words])))
    return out


def chunk_corpus(docs: dict[str, str], **kw) -> list[Chunk]:
    return [c for path, text in docs.items() for c in chunk(path, text, **kw)]


# ---------------------------------------------------------------- BM25

WORD = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase runs of letters, digits and underscores.

    `immich_postgres` and `purge_keep_days` stay one token, which is the whole
    point of BM25 here; `forwardAuth` becomes `forwardauth` in both the query
    and the docs, so it still matches.
    """
    return WORD.findall(text.lower())


class BM25:
    """Okapi BM25 over chunks.

    score(q, d) = sum over query terms t of
        idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * len(d) / avg_len))

    idf rewards rare terms (an identifier that appears in 3 chunks out of
    2,000 weighs a lot); k1 caps how much repeating a term helps; b
    penalises long chunks that match just because they are long.
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(c.text)) for c in self.chunks]
        self.len = [sum(tf.values()) for tf in self.tf]
        self.avg_len = sum(self.len) / len(self.len)
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        out = []
        for tf, length in zip(self.tf, self.len, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            out.append(
                sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            )
        return out

    def search(self, query: str) -> list[tuple[Chunk, float]]:
        return _rank(self.chunks, self.scores(query))


# ---------------------------------------------------------------- dense

Embedder = Callable[[list[str]], list[list[float]]]


def ollama_embedder(
    model: str = "bge-m3", url: str = "http://localhost:11434", batch: int = 32
) -> Embedder:
    """Embeddings from a local Ollama server (`ollama pull bge-m3` first)."""

    def embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch):
            body = json.dumps({"model": model, "input": texts[i : i + batch]}).encode()
            req = urllib.request.Request(
                f"{url}/api/embed", data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                vectors.extend(json.loads(r.read())["embeddings"])
        return vectors

    return embed


class DenseIndex:
    """Cosine similarity between the query vector and every chunk vector.

    Vectors are normalised once, so cosine is a plain dot product. Chunk
    embeddings are cached by content hash: a second run, or a run after
    changing only the queries, doesn't embed the corpus again.
    """

    def __init__(self, chunks: Sequence[Chunk], embed: Embedder, cache: Path | None = None):
        self.chunks = list(chunks)
        self.embed = embed
        vectors = self._load_or_embed(
            [c.key for c in self.chunks], [c.text for c in self.chunks], cache
        )
        self.matrix = _normalise(np.asarray(vectors, dtype=np.float32))

    def _load_or_embed(self, keys: list[str], texts: list[str], cache: Path | None) -> list:
        stored: dict[str, np.ndarray] = {}
        if cache and cache.is_file():
            with np.load(cache) as z:
                stored = {k: z[k] for k in z.files}
        missing = [i for i, k in enumerate(keys) if k not in stored]
        if missing:
            new = self.embed([texts[i] for i in missing])
            for i, v in zip(missing, new, strict=True):
                stored[keys[i]] = np.asarray(v, dtype=np.float32)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, **stored)
        return [stored[k] for k in keys]

    def scores(self, query: str) -> list[float]:
        q = _normalise(np.asarray(self.embed([query]), dtype=np.float32))[0]
        return (self.matrix @ q).tolist()

    def search(self, query: str) -> list[tuple[Chunk, float]]:
        return _rank(self.chunks, self.scores(query))


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


# ---------------------------------------------------------------- rankings


def _rank(chunks: list[Chunk], scores: list[float]) -> list[tuple[Chunk, float]]:
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [(chunks[i], scores[i]) for i in order if scores[i] > 0]


def to_doc_ranking(hits: list[tuple[Chunk, float]], depth: int) -> list[str]:
    """Documents in order of their best chunk, cut at `depth`.

    `depth` is how many candidates a retriever hands over, like the top-k a
    real RAG system retrieves. A document below it is invisible to RRF, which
    is exactly the situation where fusion can hurt.
    """
    seen: list[str] = []
    for c, _ in hits:
        if c.doc not in seen:
            seen.append(c.doc)
            if len(seen) == depth:
                break
    return seen


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank of d).

    Only positions count, never the original scores, so BM25 and cosine
    don't need to be on the same scale. A document missing from a ranking
    gets nothing from it. Ties keep the order of first appearance.
    """
    score: dict[str, float] = {}
    for ranking in rankings:
        for pos, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1 / (k + pos)
    return sorted(score, key=lambda d: -score[d])
