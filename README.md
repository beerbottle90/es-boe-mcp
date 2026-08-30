# es-boe-mcp

**Spain 🇪🇸 legal research over the Model Context Protocol.**
boe.es datos abiertos — Boletín Oficial del Estado

No authentication. No dependencies — pure Python standard library, so it runs on
a stock Python 3.9+ with nothing to install.

## Why this server exists

The consolidated-legislation API is public and free but refuses every request without `Accept: application/xml` — so an LLM's plain URL-fetch tool, which cannot set headers, sees none of it. A server can set the header. The list endpoint also has no search parameter, so search is local.

## Tools

- `search_legislation`
- `get_act`
- `get_act_text`
- `get_document`
- `get_daily_gazette`
- `server_status`

Every search response carries a `retrieval` block reporting which channels ran,
how many documents are indexed, and whether semantic search is actually on — so
a thin result set is never mistaken for a settled question.

## Search

Three retrieval channels, fused with Reciprocal Rank Fusion:

| Channel | What it is | Always available |
|---|---|---|
| `lexical` | SQLite FTS5 + BM25, diacritic-insensitive, with a strict-AND → prefix → OR ladder | yes |
| `fuzzy` | FTS5 trigram, for substrings and misspellings | yes |
| `semantic` | Dense vectors over an OpenAI-compatible embeddings endpoint | **only if configured** |

**Semantic search is opt-in and honest about it.** The standard library cannot
run a transformer, so dense retrieval needs an external endpoint:

```
EMBEDDINGS_URL=https://api.example.com/v1/embeddings
EMBEDDINGS_MODEL=text-embedding-3-small      # optional
EMBEDDINGS_API_KEY=...                       # optional
```

With nothing configured, `mode="hybrid"` degrades to lexical + fuzzy and **says
so** in every response. It never presents a keyword match as a conceptual one.
For cross-language work — a Turkish question against a Spain corpus — the
configured model must itself be multilingual; that is the operator's choice and
this server cannot verify it.

## Upstream quirks this server handles

- `Accept: application/xml` is mandatory; JSON and */* both return 400.
- The list endpoint takes only limit/offset — `?query=` returns 500.
- Consolidated text vs as-published are different endpoints; never conflate them.

## Run it

```sh
# stdio, for a desktop MCP client
python server.py

# Streamable HTTP, for a remote connector
python server.py --transport http --port 8000
# -> http://127.0.0.1:8000/mcp
```

### Build the search index first

```sh
python crawl.py --max 0
```

Direct fetches and browsing work without it; keyword search does not.

### Connecting from claude.ai

Settings → Connectors → *Add custom connector* → the `/mcp` URL → auth **None**.

> ⚠️ This server has **no authentication**. Binding it to `0.0.0.0` exposes every
> tool to anyone who can reach the port. That is why `--host` defaults to
> loopback and must be widened deliberately.

## Docker / Railway

```sh
docker build -t es-boe-mcp .
docker run -p 8000:8000 -e PORT=8000 es-boe-mcp
```

`start.sh` launches the server immediately and crawls in the background when
`CRAWL_ARGS` is set and no index exists yet, so the platform health check at
`/health` passes straight away.

## Deploying to Smithery

This repo ships `smithery.yaml`, so it can be deployed as a hosted container:

1. Push the repo to GitHub (done).
2. On [smithery.ai](https://smithery.ai), add the server from this GitHub repo.
3. Deploy. Smithery builds `Dockerfile` and serves Streamable HTTP on `/mcp`.

Optional configuration, all of it safe to leave empty:

| Variable | Effect |
|---|---|
| `CRAWL_ARGS` | What to index on first boot (default `--max 0`) |
| `EMBEDDINGS_URL` | Turns the semantic channel on |
| `EMBEDDINGS_MODEL` | Must be multilingual for cross-language search |
| `EMBEDDINGS_API_KEY` | Bearer token, if the endpoint needs one |

> Container filesystems are usually ephemeral. The index rebuilds on a cold
> start, which is why the crawl runs in the background and the server answers
> from the first second — direct fetches and browsing never need an index.

## Citation discipline

Every result carries a `citation` field built from what the source returned.
**Copy it verbatim.** Do not assemble an ECLI, a Dz.U. number, a BOE id or an
ELI by hand — malformed identifiers are rejected rather than guessed at.

## Licence

MIT. The underlying legal data belongs to its publisher and carries that
publisher's own terms.

---

Part of the [ArthurLegal](https://github.com/beerbottle90/ArthurLegal)
multi-jurisdiction legal assistant. Built in the spirit of
[saidsurucu/yargi-mcp](https://github.com/saidsurucu/yargi-mcp).
