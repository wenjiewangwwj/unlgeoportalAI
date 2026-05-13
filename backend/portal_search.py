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
        return f"{base} group:{gid}"
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
    parts: list[str] = []
    base = portal_q.strip()
    if base:
        parts.append(base)
    for t in tags:
        t = t.strip()
        if not t:
            continue
        if t.lower().startswith("tags:"):
            parts.append(t)
        else:
            parts.append(f"tags:{t}")
    return " ".join(parts).strip()
