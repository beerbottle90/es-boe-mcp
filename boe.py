"""Client for Spain's BOE open-data API — standard library only.

Why this server exists
----------------------
The consolidated-legislation API is fully public and free, but it refuses every
request that does not carry ``Accept: application/xml``:

    Accept: application/xml   -> 200  (Ley de Sociedades de Capital)
    Accept: application/json  -> 400  "No soportado ningún mime type"
    no Accept header          -> 400

An LLM's plain URL-fetch tool cannot set request headers, so the whole
consolidated corpus is invisible to it. A server can set the header, which is
the entire reason this wrapper exists.

The second reason: the list endpoint takes only ``limit`` and ``offset`` —
there is **no title or full-text search parameter** (``?query=`` returns 500,
``?titulo=`` returns 400). Search therefore has to be built locally, which is
what ``crawl.py`` and the shared ``retrieval`` index do.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

__version__ = "1.0.0"

BASE = "https://www.boe.es"
API = BASE + "/datosabiertos/api"
UA = "arthurlegal-es-boe-mcp/%s (+https://github.com/beerbottle90/es-boe-mcp)" % __version__

# The BOE ids this server accepts. Anything else is a caller mistake, and
# validating here keeps a malformed id from becoming a confusing upstream 404.
BOE_ID = re.compile(r"^BOE-[A-Z]-\d{4}-\d+$")


class BoeError(Exception):
    """An upstream failure worth explaining to the caller."""


def _get(url: str, accept: str = "application/xml", timeout: int = 45) -> bytes:
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", accept)          # <- the point of this whole module
    req.add_header("User-Agent", UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:400]
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
        if exc.code == 404:
            raise BoeError("Not found (404): %s" % url) from exc
        if exc.code == 400 and "mime" in body.lower():
            raise BoeError(
                "BOE rejected the Accept header — this endpoint only speaks "
                "application/xml. %s" % body
            ) from exc
        raise BoeError("HTTP %s from BOE: %s %s" % (exc.code, url, body)) from exc
    except urllib.error.URLError as exc:
        raise BoeError("Could not reach BOE: %s" % exc.reason) from exc


def _text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _iso(compact: str) -> str:
    """BOE dates are ``YYYYMMDD`` (sometimes with a time suffix); make them ISO."""
    digits = re.sub(r"\D", "", compact or "")[:8]
    if len(digits) != 8:
        return ""
    return "%s-%s-%s" % (digits[0:4], digits[4:6], digits[6:8])


def _parse_root(raw: bytes) -> ET.Element:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise BoeError("BOE returned unparseable XML: %s" % exc) from exc
    status = root.find("./status/code")
    if status is not None and _text(status) not in ("200", ""):
        raise BoeError("BOE status %s: %s" % (_text(status), _text(root.find("./status/text"))))
    return root


class BoeClient:
    """Thin, honest wrapper. Every method names the endpoint it calls."""

    # -- consolidated legislation ---------------------------------------- #
    def list_consolidated(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """``GET /legislacion-consolidada?limit&offset``.

        Metadata only, no search parameters — pagination is the *only* way
        through the corpus, which is why the crawler exists.
        """
        limit = max(1, min(int(limit), 1000))     # 1000 verified as the working ceiling
        url = "%s/legislacion-consolidada?limit=%d&offset=%d" % (API, limit, int(offset))
        root = _parse_root(_get(url))
        out = []
        for item in root.findall("./data/item"):
            ident = _text(item.find("identificador"))
            # 'S' means the act's validity is spent — BOE's own repeal signal.
            # Carried through to the index so search results can say so.
            spent = _text(item.find("vigencia_agotada")).upper() == "S"
            out.append(
                {
                    "id": ident,
                    "title": _text(item.find("titulo")),
                    "rango": _text(item.find("rango")),
                    "departamento": _text(item.find("departamento")),
                    "ambito": _text(item.find("ambito")),
                    "numero_oficial": _text(item.find("numero_oficial")),
                    "date": _iso(_text(item.find("fecha_disposicion"))),
                    "published": _iso(_text(item.find("fecha_publicacion"))),
                    "updated": _text(item.find("fecha_actualizacion")),
                    "fecha_vigencia": _iso(_text(item.find("fecha_vigencia"))),
                    "vigencia_agotada": spent,
                    "status": "vigencia agotada" if spent else "vigente",
                    "estado_consolidacion": _text(item.find("estado_consolidacion")),
                    "eli": _text(item.find("url_eli")),
                    "url": _text(item.find("url_html_consolidada"))
                           or "%s/buscar/act.php?id=%s" % (BASE, ident),
                }
            )
        return out

    def get_consolidated(self, boe_id: str, part: str = "") -> Dict[str, Any]:
        """``GET /legislacion-consolidada/id/{id}[/metadatos|/analisis|/texto]``."""
        boe_id = (boe_id or "").strip().upper()
        if not BOE_ID.match(boe_id):
            raise BoeError("Malformed BOE id %r — expected e.g. BOE-A-2010-10544" % boe_id)
        part = (part or "").strip("/")
        if part and part not in ("metadatos", "analisis", "texto"):
            raise BoeError("part must be one of: metadatos, analisis, texto")
        url = "%s/legislacion-consolidada/id/%s%s" % (API, boe_id, "/" + part if part else "")
        root = _parse_root(_get(url))
        data = root.find("./data")
        if data is None:
            raise BoeError("No <data> in BOE response for %s" % boe_id)

        meta = data.find("metadatos") if data.find("metadatos") is not None else data
        result: Dict[str, Any] = {
            "id": boe_id,
            "title": _text(meta.find("titulo")),
            "rango": _text(meta.find("rango")),
            "departamento": _text(meta.find("departamento")),
            "date": _iso(_text(meta.find("fecha_disposicion"))),
            "published": _iso(_text(meta.find("fecha_publicacion"))),
            # These two are the status discipline: BOE says outright whether the
            # act has been repealed or annulled. Never report a text as being in
            # force without echoing them.
            "estatus_derogacion": _text(meta.find("estatus_derogacion")),
            "estatus_anulacion": _text(meta.find("estatus_anulacion")),
            "fecha_vigencia": _iso(_text(meta.find("fecha_vigencia"))),
            "url": "%s/buscar/act.php?id=%s" % (BASE, boe_id),
            "citation": "%s (%s)" % (_text(meta.find("titulo")), boe_id),
        }
        materias = [_text(m) for m in data.findall(".//materias/materia")]
        if materias:
            result["materias"] = materias
        text_el = data.find("texto")
        if text_el is not None:
            result["text"] = _text(text_el)
        return result

    def get_text(self, boe_id: str, max_chars: int = 60000) -> Dict[str, Any]:
        """Consolidated full text. Long acts are truncated with an explicit marker."""
        doc = self.get_consolidated(boe_id, part="texto")
        body = doc.get("text") or ""
        doc["length_chars"] = len(body)
        if len(body) > max_chars:
            doc["text"] = body[:max_chars]
            doc["truncated"] = (
                "Text truncated at %d of %d characters. Request a specific article "
                "or raise max_chars." % (max_chars, len(body))
            )
        return doc

    # -- daily gazette ---------------------------------------------------- #
    def daily_summary(self, date_yyyymmdd: str) -> Dict[str, Any]:
        """``GET /boe/sumario/{YYYYMMDD}`` — what was published that day."""
        digits = re.sub(r"\D", "", date_yyyymmdd or "")
        if len(digits) != 8:
            raise BoeError("date must be YYYYMMDD or YYYY-MM-DD, got %r" % date_yyyymmdd)
        root = _parse_root(_get("%s/boe/sumario/%s" % (API, digits)))
        items = []
        for it in root.findall(".//item"):
            ident = _text(it.find("identificador"))
            if not ident:
                continue
            items.append(
                {
                    "id": ident,
                    "title": _text(it.find("titulo")),
                    "url": "%s/diario_boe/xml.php?id=%s" % (BASE, ident),
                }
            )
        return {"date": _iso(digits), "count": len(items), "items": items}

    # -- single published document ---------------------------------------- #
    def get_document(self, boe_id: str, max_chars: int = 60000) -> Dict[str, Any]:
        """``GET /diario_boe/xml.php?id={id}`` — the act **as published**.

        This is *not* the consolidated text: later amendments are not applied.
        The returned ``consolidated: False`` flag exists so a caller cannot
        mistake one for the other.
        """
        boe_id = (boe_id or "").strip().upper()
        if not BOE_ID.match(boe_id):
            raise BoeError("Malformed BOE id %r" % boe_id)
        raw = _get("%s/diario_boe/xml.php?id=%s" % (BASE, urllib.parse.quote(boe_id)))
        root = _parse_root(raw)
        meta = root.find("./metadatos")
        body = _text(root.find("./texto"))
        out = {
            "id": boe_id,
            "consolidated": False,
            "warning": "Text as published in the BOE. Later amendments are NOT "
                       "applied — use get_consolidated_act for the in-force text.",
            "title": _text(meta.find("titulo")) if meta is not None else "",
            "departamento": _text(meta.find("departamento")) if meta is not None else "",
            "rango": _text(meta.find("rango")) if meta is not None else "",
            "date": _iso(_text(meta.find("fecha_disposicion"))) if meta is not None else "",
            "url": "%s/diario_boe/xml.php?id=%s" % (BASE, boe_id),
            "length_chars": len(body),
            "text": body[:max_chars],
        }
        if len(body) > max_chars:
            out["truncated"] = "Truncated at %d of %d characters." % (max_chars, len(body))
        return out
