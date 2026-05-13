import json
import re
from typing import Any

import httpx

from backend.config import settings

SYSTEM_PROMPT = """You help users find datasets in an ArcGIS Portal catalog.
Given a natural-language question, output a JSON object ONLY (no markdown) with:
- "portal_q": a short search string for ArcGIS Portal /sharing/rest/search parameter q.
  Use relevant synonyms (e.g. population, census, ACS, demographics). Under 220 characters.
- "tags": array of 0 to 5 tag strings without "tags:" prefix, or empty array.
- "user_note": one short sentence explaining what you search for (for the UI).

ArcGIS q can combine words; optional filters look like tags:water or type:"Feature Service".
Prefer plain keywords unless a filter clearly helps."""


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("LLM did not return a JSON object")
    return json.loads(m.group())


async def expand_query_natural_language(user_text: str) -> dict[str, Any]:
    user_text = user_text.strip()
    if not user_text:
        return {"portal_q": "", "tags": [], "user_note": ""}

    provider = settings.llm_provider.lower().strip()
    if provider == "gemini":
        raw = await _gemini_chat(user_text)
    elif provider == "ollama":
        raw = await _ollama_chat(user_text)
    elif provider == "openai_compatible":
        raw = await _openai_compatible_chat(user_text)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")

    data = _parse_json_object(raw)
    portal_q = str(data.get("portal_q", "")).strip()
    tags = data.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if str(t).strip()][:5]
    user_note = str(data.get("user_note", "")).strip()
    return {"portal_q": portal_q, "tags": tags, "user_note": user_note}


async def _ollama_chat(user_text: str) -> str:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        body = r.json()
    msg = body.get("message") or {}
    return msg.get("content") or ""


async def _gemini_chat(user_text: str) -> str:
    key = settings.gemini_api_key.strip()
    if not key:
        raise ValueError("GEMINI_API_KEY must be set (get a free key at https://aistudio.google.com/apikey )")

    model = (settings.gemini_model or "gemini-2.0-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, params={"key": key}, json=body)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"raw": r.text[:500]}
            raise ValueError(f"Gemini HTTP {r.status_code}: {err}")
        data = r.json()

    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError(f"Gemini returned no candidates: {data}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts: list[str] = []
    for p in parts:
        t = p.get("text")
        if t:
            texts.append(str(t))
    return "\n".join(texts).strip()


async def _openai_compatible_chat(user_text: str) -> str:
    if not settings.openai_compat_base_url or not settings.openai_compat_api_key:
        raise ValueError("OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_API_KEY must be set")

    url = f"{settings.openai_compat_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.openai_compat_api_key}"}
    payload = {
        "model": settings.openai_compat_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        body = r.json()
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return msg.get("content") or ""
