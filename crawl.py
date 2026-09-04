"""Build the local search index for es-boe-mcp.

    python crawl.py            # full metadata crawl (~20 requests)
    python crawl.py --embed    # ...then compute vectors, if EMBEDDINGS_URL is set

What gets indexed, and why only this
------------------------------------
Title, official number, issuing department, rango, dates and ámbito — **not**
the consolidated body text, and **not** ``materias``. BOE exposes its subject
tags only on the per-act detail endpoint, so indexing them would cost one
request per act (12,376 of them) rather than the ~20 this crawl takes. The
subject signal therefore comes from the title, which for Spanish legislation is
descriptive by convention ("Ley 17/2006, de control ambiental integrado").

That is a deliberate trade, not an oversight. A single consolidated act runs to
roughly a million characters (the Ley de Sociedades de Capital is 960,963), so
indexing bodies for the whole corpus would mean tens of gigabytes for a server
whose job is to *find the act*. Spanish legislation is looked up by name,
number and subject; once the act is identified, ``get_act_text`` fetches the
authoritative text on demand.

The consequence is stated in every search response so it cannot be mistaken:
this searches metadata, so a phrase buried in article 348 bis will not be found
by searching for it.
"""

from __future__ import annotations

import argparse
import sys
import time

from boe import BoeClient, BoeError
from retrieval import Index, embeddings_available

PAGE = 1000  # verified working ceiling for the list endpoint


def crawl(index: Index, max_items: int = 0, pause: float = 0.3) -> int:
    client = BoeClient()
    offset, seen = 0, 0
    while True:
        try:
            batch = client.list_consolidated(limit=PAGE, offset=offset)
        except BoeError as exc:
            sys.stderr.write("stopped at offset %d: %s\n" % (offset, exc))
            break
        if not batch:
            break
        for item in batch:
            index.upsert(
                {
                    "ref": item["id"],
                    "title": item["title"],
                    # The searchable surface: everything that identifies the act.
                    "body": "\n".join(
                        x for x in (
                            item.get("numero_oficial"),
                            item.get("rango"),
                            item.get("departamento"),
                            item.get("ambito"),
                        ) if x
                    ),
                    "url": item["url"],
                    "lang": "es",
                    "date": item.get("date") or item.get("published") or "",
                    "court": item.get("departamento") or "",
                    "subject": item.get("ambito") or "",
                    # BOE's own repeal signal, indexed so search can filter on it.
                    "status": item.get("status") or "",
                    "citation": "%s (%s)" % (item["title"], item["id"]),
                    "meta": {
                        "rango": item.get("rango"),
                        "numero_oficial": item.get("numero_oficial"),
                        "published": item.get("published"),
                        "fecha_vigencia": item.get("fecha_vigencia"),
                        "estado_consolidacion": item.get("estado_consolidacion"),
                        "eli": item.get("eli"),
                    },
                }
            )
            seen += 1
        index.db.commit()
        sys.stderr.write("indexed %d\r" % seen)
        sys.stderr.flush()
        if max_items and seen >= max_items:
            break
        if len(batch) < PAGE:
            break
        offset += PAGE
        time.sleep(pause)  # be a polite client of a public service
    index.reindex_fts()
    index.set_state("last_crawl", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    index.set_state("corpus", "BOE legislación consolidada (metadata)")
    sys.stderr.write("\nindexed %d documents\n" % seen)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the es-boe-mcp index")
    ap.add_argument("--max", type=int, default=0, help="stop after N acts (0 = all)")
    ap.add_argument("--embed", action="store_true", help="compute vectors after crawling")
    ap.add_argument("--index", default=None, help="index path (default $INDEX_PATH or index.db)")
    args = ap.parse_args()

    index = Index(args.index)
    crawl(index, max_items=args.max)
    if args.embed:
        if not embeddings_available():
            sys.stderr.write("EMBEDDINGS_URL not set — skipping vectors.\n")
        else:
            sys.stderr.write("%s\n" % index.embed_missing())


if __name__ == "__main__":
    main()
