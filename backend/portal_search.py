import re
from typing import Any
from urllib.parse import urlencode

import httpx

from backend.config import settings


def apply_group_scope(q: str) -> str:
    """Append ArcGIS Portal group filter to q (items shared with that group)."""
    gid = (settings.portal_group_id or "").strip()
    if not gid:
        return q.strip()
    base = q.strip()
    if base:
        return f"({base}) group:{gid}" if (" OR " in base or " AND " in base) else f"{base} group:{gid}"
    return f"group:{gid}"


def build_search_url(q: str, num: int = 20, start: int = 1) -> str:
    base = f"{settings.sharing_rest}/search"
    params: dict[str, Any] = {"f": "json", "q": q, "num": num, "start": start}
    if settings.portal_token:
        params["token"] = settings.portal_token
    return f"{base}?{urlencode(params)}"


async def portal_search(q: str, num: int = 20, start: int = 1) -> dict[str, Any]:
    q_scoped = apply_group_scope(q)
    url = build_search_url(q_scoped, num=num, start=start)
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


def merge_tags_into_q(portal_q: str, tags: list[str]) -> str:
    def clean_term(term: str) -> str:
        term = term.strip()
        term = term.removeprefix("tags:").removeprefix("tag:").strip()
        term = term.strip("\"'")
        return re.sub(r"\s+", " ", term)

    def quote_term(term: str) -> str:
        term = clean_term(term)
        if not term:
            return ""
        if " " in term and not (term.startswith('"') and term.endswith('"')):
            return f'"{term}"'
        return term

    def add_term(bucket: list[str], seen: set[str], term: str) -> None:
        term = quote_term(term)
        if not term:
            return
        key = term.lower()
        if key in seen:
            return
        seen.add(key)
        bucket.append(term)

    def expand_synonyms(text: str) -> list[str]:
        lowered = text.lower()
        extras: list[str] = []
        if any(word in lowered for word in ("farmland", "agriculture", "cropland", "farming", "farm")):
            extras.extend(["agriculture", "cropland", "land cover", "land use", "NLCD"])
        if any(word in lowered for word in ("land cover", "landuse", "land use", "nlcd")):
            extras.extend(["land cover", "land use", "NLCD"])
        return extras

    parts: list[str] = []
    seen: set[str] = set()

    base = clean_term(portal_q)
    if base:
        add_term(parts, seen, base)
        for extra in expand_synonyms(base):
            add_term(parts, seen, extra)

    for t in tags:
        t = clean_term(t)
        if not t:
            continue
        add_term(parts, seen, t)
        for extra in expand_synonyms(t):
            add_term(parts, seen, extra)

    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "(" + " OR ".join(parts) + ")"
