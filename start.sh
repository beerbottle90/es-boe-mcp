#!/bin/sh
# Start the MCP server immediately, and build the search index in the background
# if it is missing.
#
# The server is useful from the first second: direct fetches and browsing work
# with no index at all, and every search tool reports an empty index plainly
# rather than returning a misleadingly empty result set. Blocking startup on a
# crawl would instead fail the platform health check.
set -e

INDEX_PATH="${INDEX_PATH:-/app/data/index.db}"
export INDEX_PATH

if [ -n "${CRAWL_ARGS}" ] && [ ! -f "${INDEX_PATH}" ]; then
    echo "no index at ${INDEX_PATH} — crawling in background: ${CRAWL_ARGS}" >&2
    # shellcheck disable=SC2086
    ( python crawl.py ${CRAWL_ARGS} || echo "crawl failed; serving without an index" >&2 ) &
fi

exec python server.py
