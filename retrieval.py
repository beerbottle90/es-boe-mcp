"""Hybrid retrieval over a local SQLite index — standard library only.

Vendored into each ArthurLegal jurisdiction MCP server. Exists because several
official legal APIs return documents but cannot *search* them:

- NL Rechtspraak: 3.75M decisions, **no free-text parameter at all**
- IE Irish Statute Book / LU Legilux: perfect document URLs, **no search endpoint**
- PL Sejm: searches **titles only**, never the body

Three retrieval channels, fused with Reciprocal Rank Fusion:

1. ``lexical``   FTS5 + BM25, ``unicode61 remove_diacritics 2`` — exact legal
                 terms, article numbers, party names. Diacritic-insensitive, so a
                 Turkish-keyboard query still matches ``Kündigung``.
2. ``fuzzy``     FTS5 ``trigram`` — substring and misspelling tolerance
                 (``nergiew`` finds ``energiewet``). Only consulted when the
                 lexical channel is thin.
3. ``semantic``  Dense vectors, **only when an embeddings backend is configured**
                 (see below). Finds conceptually related text that shares no
                 keywords — the actual point of asking a question in Turkish
                 about a Dutch judgment.

Honest limitation
-----------------
The standard library cannot run a transformer. ``semantic`` therefore requires
an OpenAI-compatible embeddings endpoint supplied through the environment:

    EMBEDDINGS_URL=https://api.example.com/v1/embeddings
    EMBEDDINGS_MODEL=text-embedding-3-small     # optional
    EMBEDDINGS_API_KEY=...                      # optional
    EMBEDDINGS_DIM=1536                         # optional, for sanity checks

With no backend configured, ``mode="hybrid"`` silently degrades to
lexical+fuzzy and **says so** in the response's ``retrieval`` block. It never
pretends a lexical hit was a semantic one.

For cross-language work (Turkish question, Dutch corpus) the configured model
must itself be multilingual; a monolingual English model will score poorly and
the results block will still claim ``semantic: on``. That is the operator's
choice, not something this module can detect.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Index", "embeddings_available", "embeddings_status"]

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS docs (
    id       INTEGER PRIMARY KEY,
    ref      TEXT UNIQUE NOT NULL,   -- stable source identifier (ECLI, ELI, BOE-ID...)
    title    TEXT NOT NULL DEFAULT '',
    body     TEXT NOT NULL DEFAULT '',
    url      TEXT NOT NULL DEFAULT '',
    lang     TEXT NOT NULL DEFAULT '',
    date     TEXT NOT NULL DEFAULT '',   -- ISO 8601, sorts lexically
    status   TEXT NOT NULL DEFAULT '',   -- in-force / repealed / publisher label
    court    TEXT NOT NULL DEFAULT '',   -- or issuing authority
    subject  TEXT NOT NULL DEFAULT '',
    citation TEXT NOT NULL DEFAULT '',   -- verbatim citation string, never built by the model
    meta     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS docs_date  ON docs(date);
CREATE INDEX IF NOT EXISTS docs_court ON docs(court);

CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    title, body, citation,
    content='docs', content_rowid='id',
    tokenize="unicode61 remove_diacritics 2"
);

CREATE VIRTUAL TABLE IF NOT EXISTS docs_tri USING fts5(
    title, body,
    content='docs', content_rowid='id',
    tokenize="trigram"
);

CREATE TABLE IF NOT EXISTS vecs (
    doc_id INTEGER PRIMARY KEY REFERENCES docs(id) ON DELETE CASCADE,
    dim    INTEGER NOT NULL,
    vec    BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS state (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""

# FTS5 query syntax characters we must not pass through from user input.
_FTS_UNSAFE = re.compile(r'["\'\(\)\*\^:{}\[\]]')


def _fts_query(raw: str, prefix: bool = False, join: str = "AND") -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Quoted phrases survive as phrases; bare words are joined by ``join``.
    Everything else is stripped, because an unescaped ``(`` or ``"`` is an FTS5
    syntax error, not a search for a bracket.

    ``prefix`` appends FTS5's ``*`` operator to bare words. That matters far more
    than it looks: Dutch, Finnish, German and Polish compound aggressively, so a
    search for ``kartel`` does not match ``kartelverbod`` under whole-word
    tokenisation, while ``kartel*`` does.
    """
    phrases = re.findall(r'"([^"]+)"', raw)
    rest = re.sub(r'"[^"]+"', " ", raw)
    words = [w for w in _FTS_UNSAFE.sub(" ", rest).split() if len(w) > 1]
    parts = ['"%s"' % p.replace('"', "") for p in phrases if p.strip()]
    # A prefix token must sit outside the quotes: "kartel"* is the valid form.
    parts += ['"%s"%s' % (w, "*" if prefix else "") for w in words]
    return (" %s " % join).join(parts)


# --------------------------------------------------------------------------- #
# Embeddings backend (optional)                                                #
# --------------------------------------------------------------------------- #
def embeddings_available() -> bool:
    return bool(os.environ.get("EMBEDDINGS_URL"))


def embeddings_status() -> Dict[str, Any]:
    if not embeddings_available():
        return {
            "semantic": "off",
            "reason": "EMBEDDINGS_URL not set — hybrid search degraded to "
                      "lexical + fuzzy. Results are keyword matches, not "
                      "conceptual matches.",
        }
    return {
        "semantic": "on",
        "model": os.environ.get("EMBEDDINGS_MODEL", "(server default)"),
        "note": "Cross-language retrieval only works if this model is multilingual.",
    }


def _embed(texts: Sequence[str], timeout: int = 60) -> List[List[float]]:
    """Call an OpenAI-compatible /v1/embeddings endpoint. Raises on failure."""
    url = os.environ.get("EMBEDDINGS_URL")
    if not url:
        raise RuntimeError("EMBEDDINGS_URL is not set")
    payload: Dict[str, Any] = {"input": list(texts)}
    model = os.environ.get("EMBEDDINGS_MODEL")
    if model:
        payload["model"] = model
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    key = os.environ.get("EMBEDDINGS_API_KEY")
    if key:
        req.add_header("Authorization", "Bearer %s" % key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    items = data.get("data") or []
    out = [it["embedding"] for it in items]
    if len(out) != len(texts):
        raise RuntimeError(
            "embeddings backend returned %d vectors for %d inputs" % (len(out), len(texts))
        )
    return out


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack("<%df" % len(vec), *vec)


def _unpack(blob: bytes, dim: int) -> Tuple[float, ...]:
    return struct.unpack("<%df" % dim, blob)


def _normalise(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# --------------------------------------------------------------------------- #
# Index                                                                        #
# --------------------------------------------------------------------------- #
class Index:
    """A local SQLite corpus with hybrid retrieval.

    ``path`` defaults to ``$INDEX_PATH`` then ``./index.db``. Concurrent readers
    are fine (WAL); a single writer is assumed, which matches the crawl-then-serve
    lifecycle these servers use.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.environ.get("INDEX_PATH") or "index.db"
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self.db.commit()

    # -- writing ---------------------------------------------------------- #
    def upsert(self, doc: Dict[str, Any]) -> int:
        """Insert or update one document by its ``ref``. Returns the row id.

        Deliberately does **not** touch the FTS tables: they are external-content
        and get rebuilt wholesale by :meth:`reindex_fts` once a crawl finishes.
        Call that before serving, or searches will miss everything just written.
        """
        cols = ("ref", "title", "body", "url", "lang", "date", "status", "court", "subject", "citation")
        values = [str(doc.get(c) or "") for c in cols]
        meta = json.dumps(doc.get("meta") or {}, ensure_ascii=False)
        row = self.db.execute("SELECT id FROM docs WHERE ref = ?", (values[0],)).fetchone()
        if row:
            doc_id = row["id"]
            self.db.execute(
                "UPDATE docs SET title=?,body=?,url=?,lang=?,date=?,status=?,court=?,"
                "subject=?,citation=?,meta=? WHERE id=?",
                values[1:] + [meta, doc_id],
            )
            # The stored vector described the old body; drop it so embed_missing
            # recomputes one rather than leaving a stale embedding behind.
            self.db.execute("DELETE FROM vecs WHERE doc_id = ?", (doc_id,))
            return doc_id
        cur = self.db.execute(
            "INSERT INTO docs(ref,title,body,url,lang,date,status,court,subject,citation,meta) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            values + [meta],
        )
        return int(cur.lastrowid)

    def reindex_fts(self) -> None:
        """Rebuild both FTS tables from ``docs``.

        External-content FTS5 tables need explicit syncing. Rebuilding wholesale
        after a crawl is far simpler — and for these corpus sizes, faster — than
        maintaining per-row delete/insert triggers.
        """
        self.db.execute("INSERT INTO docs_fts(docs_fts) VALUES('rebuild')")
        self.db.execute("INSERT INTO docs_tri(docs_tri) VALUES('rebuild')")
        self.db.commit()

    def embed_missing(self, batch: int = 64, limit: Optional[int] = None) -> Dict[str, Any]:
        """Compute and store vectors for documents that lack one."""
        if not embeddings_available():
            return {"embedded": 0, **embeddings_status()}
        sql = (
            "SELECT d.id, d.title, d.body FROM docs d "
            "LEFT JOIN vecs v ON v.doc_id = d.id WHERE v.doc_id IS NULL"
        )
        if limit:
            sql += " LIMIT %d" % int(limit)
        rows = self.db.execute(sql).fetchall()
        done = 0
        for i in range(0, len(rows), batch):
            chunk = rows[i : i + batch]
            texts = [
                ((r["title"] or "") + "\n" + (r["body"] or ""))[:8000] for r in chunk
            ]
            vectors = _embed(texts)
            for row, vec in zip(chunk, vectors):
                unit = _normalise(vec)
                self.db.execute(
                    "INSERT OR REPLACE INTO vecs(doc_id, dim, vec) VALUES(?,?,?)",
                    (row["id"], len(unit), _pack(unit)),
                )
            self.db.commit()
            done += len(chunk)
        return {"embedded": done, **embeddings_status()}

    def set_state(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO state(k,v) VALUES(?,?)", (key, value))
        self.db.commit()

    def get_state(self, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT v FROM state WHERE k = ?", (key,)).fetchone()
        return row["v"] if row else default

    def count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) AS n FROM docs").fetchone()["n"])

    # -- reading ---------------------------------------------------------- #
    def _where(self, filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
        clauses, params = [], []
        for col in ("lang", "court", "status", "subject"):
            val = filters.get(col)
            if val:
                clauses.append("d.%s = ?" % col)
                params.append(val)
        if filters.get("date_from"):
            clauses.append("d.date >= ?")
            params.append(filters["date_from"])
        if filters.get("date_to"):
            clauses.append("d.date <= ?")
            params.append(filters["date_to"])
        return (" AND " + " AND ".join(clauses) if clauses else ""), params

    def _run_fts(self, expr: str, filters: Dict[str, Any], k: int) -> List[int]:
        if not expr:
            return []
        where, params = self._where(filters)
        sql = (
            "SELECT d.id FROM docs_fts f JOIN docs d ON d.id = f.rowid "
            "WHERE docs_fts MATCH ?" + where + " ORDER BY bm25(docs_fts, 4.0, 1.0, 2.0) LIMIT ?"
        )
        try:
            rows = self.db.execute(sql, [expr] + params + [k]).fetchall()
        except sqlite3.OperationalError:
            # A malformed MATCH expression is a bad query, not a server fault.
            return []
        return [r["id"] for r in rows]

    def _lexical(self, query: str, filters: Dict[str, Any], k: int) -> List[int]:
        """Strict AND first, then progressively looser, stopping as soon as it bites.

        Legal corpora punish a single strategy: an exact multi-term AND is right
        when the user knows the terminology and empty when they are one compound
        boundary off. The ladder keeps precision when precision is available and
        degrades to recall only when the stricter rung returned nothing.
        """
        ids = self._run_fts(_fts_query(query), filters, k)
        if ids:
            return ids
        ids = self._run_fts(_fts_query(query, prefix=True), filters, k)
        if ids:
            return ids
        return self._run_fts(_fts_query(query, prefix=True, join="OR"), filters, k)

    def _fuzzy(self, query: str, filters: Dict[str, Any], k: int) -> List[int]:
        # trigram needs a contiguous string of >= 3 chars; use the longest word.
        words = [w for w in re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE).split() if len(w) >= 3]
        if not words:
            return []
        needle = max(words, key=len)
        where, params = self._where(filters)
        sql = (
            "SELECT d.id FROM docs_tri f JOIN docs d ON d.id = f.rowid "
            "WHERE docs_tri MATCH ?" + where + " LIMIT ?"
        )
        try:
            rows = self.db.execute(sql, ['"%s"' % needle] + params + [k]).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r["id"] for r in rows]

    def _semantic(self, query: str, filters: Dict[str, Any], k: int, scan_max: int) -> List[int]:
        if not embeddings_available():
            return []
        try:
            qvec = _normalise(_embed([query])[0])
        except (urllib.error.URLError, RuntimeError, KeyError, ValueError):
            return []
        where, params = self._where(filters)
        rows = self.db.execute(
            "SELECT v.doc_id AS id, v.dim, v.vec FROM vecs v JOIN docs d ON d.id = v.doc_id "
            "WHERE 1=1" + where + " LIMIT ?",
            params + [scan_max],
        ).fetchall()
        scored = []
        qlen = len(qvec)
        for r in rows:
            if r["dim"] != qlen:
                continue
            vec = _unpack(r["vec"], r["dim"])
            # Both sides are unit vectors, so the dot product is the cosine.
            scored.append((sum(a * b for a, b in zip(qvec, vec)), r["id"]))
        scored.sort(reverse=True)
        return [doc_id for _, doc_id in scored[:k]]

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        snippet_chars: int = 320,
    ) -> Dict[str, Any]:
        """Retrieve documents. ``mode``: hybrid | lexical | semantic | fuzzy."""
        filters = filters or {}
        pool = max(limit * 5, 50)
        scan_max = int(os.environ.get("SEMANTIC_SCAN_MAX", "50000"))

        channels: Dict[str, List[int]] = {}
        if mode in ("hybrid", "lexical"):
            channels["lexical"] = self._lexical(query, filters, pool)
        if mode in ("hybrid", "semantic"):
            channels["semantic"] = self._semantic(query, filters, pool, scan_max)
        if mode == "fuzzy" or (mode == "hybrid" and len(channels.get("lexical") or []) < limit):
            channels["fuzzy"] = self._fuzzy(query, filters, pool)

        # Reciprocal Rank Fusion: rank-based, so channels with incomparable score
        # scales (BM25 is unbounded and negative, cosine is [-1,1]) combine safely.
        rrf_k = 60
        weights = {"lexical": 1.0, "semantic": 1.0, "fuzzy": 0.4}
        fused: Dict[int, float] = {}
        for channel, ids in channels.items():
            w = weights.get(channel, 1.0)
            for rank, doc_id in enumerate(ids):
                fused[doc_id] = fused.get(doc_id, 0.0) + w / (rrf_k + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        results = []
        for doc_id, score in ordered:
            row = self.db.execute(
                "SELECT ref,title,url,lang,date,status,court,subject,citation,body,meta "
                "FROM docs WHERE id = ?",
                (doc_id,),
            ).fetchone()
            if row is None:
                continue
            body = row["body"] or ""
            results.append(
                {
                    "ref": row["ref"],
                    "title": row["title"],
                    "citation": row["citation"],
                    "url": row["url"],
                    "date": row["date"],
                    "status": row["status"],
                    "court": row["court"],
                    "lang": row["lang"],
                    "score": round(score, 6),
                    "snippet": body[:snippet_chars] + ("…" if len(body) > snippet_chars else ""),
                }
            )

        retrieval = {
            "mode": mode,
            "channels_used": {k: len(v) for k, v in channels.items()},
            "indexed_documents": self.count(),
            **embeddings_status(),
        }
        if mode in ("hybrid", "semantic") and not embeddings_available():
            retrieval["warning"] = (
                "Semantic channel unavailable — these are keyword matches. A "
                "conceptually related document that shares no keywords was NOT "
                "retrieved and may still exist."
            )
        return {"query": query, "total": len(results), "retrieval": retrieval, "results": results}

    def get(self, ref: str) -> Optional[Dict[str, Any]]:
        row = self.db.execute(
            "SELECT ref,title,body,url,lang,date,status,court,subject,citation,meta "
            "FROM docs WHERE ref = ?",
            (ref,),
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        try:
            out["meta"] = json.loads(out.get("meta") or "{}")
        except ValueError:
            out["meta"] = {}
        return out
