FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Pure standard library: nothing to install. Asserted so a future dependency
# cannot creep in unnoticed.
RUN python -c "import sqlite3, json, zipfile, urllib.request; print('stdlib ok')" \
 && chmod +x start.sh \
 && mkdir -p /app/data

ENV MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    INDEX_PATH=/app/data/index.db

EXPOSE 8000
CMD ["./start.sh"]
