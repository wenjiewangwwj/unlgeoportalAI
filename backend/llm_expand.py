import json
import re
from collections.abc import Awaitable, Callable
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


def _passthrough(user_text: str, user_note: str, llm_mode: str = "none") -> dict[str, Any]:
    return {
        "portal_q": user_text,
        "tags": [],
        "user_note": user_note,
        "llm_used": False,
        "llm_mode": llm_mode,
    }


def _normalize_llm_payload(data: dict[str, Any], llm_mode: str) -> dict[str, Any]:
    portal_q = str(data.get("portal_q", "")).strip()
    tags = data.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if str(t).strip()][:5]
    user_note = str(data.get("user_note", "")).strip()
    return {
        "portal_q": portal_q,
        "tags": tags,
        "user_note": user_note,
        "llm_used": True,
        "llm_mode": llm_mode,
    }


async def _try_llm_json(llm_mode: str, fetch_text: Callable[[], Awaitable[str]]) -> dict[str, Any] | None:
    try:
        raw = (await fetch_text()).strip()
        if not raw:
            return None
        data = _parse_json_object(raw)
        out = _normalize_llm_payload(data, llm_mode)
        if not out["portal_q"] and not out["tags"]:
            return None
        return out
    except Exception:
        return None


async def expand_query_natural_language(user_text: str) -> dict[str, Any]:
    user_text = user_text.strip()
    if not user_text:
        return _passthrough("", "", "none")

    prov = settings.llm_provider.lower().strip()

    if prov in ("none", "off", "disabled"):
        return _passthrough(user_text, "LLM disabled (LLM_PROVIDER=none). Portal search uses your words as-is.", "none")

    if prov == "auto":
        if settings.gemini_api_key.strip():
            out = await _try_llm_json("gemini", lambda: _gemini_chat(user_text))
            if out:
                return out
        out = await _try_llm_json("huggingface", lambda: _huggingface_chat(user_text))
        if out:
            return out
        return _passthrough(
            user_text,
            "No LLM expansion: add a free Gemini key (GEMINI_API_KEY) and/or a free Hugging Face token "
            "(HUGGINGFACE_API_KEY from https://huggingface.co/settings/tokens ). Searching with your text as-is.",
            "none",
        )

    if prov == "gemini":
        if not settings.gemini_api_key.strip():
            return _passthrough(
                user_text,
                "No GEMINI_API_KEY set; skipping Gemini. Searching with your text as-is.",
                "none",
            )
        out = await _try_llm_json("gemini", lambda: _gemini_chat(user_text))
        return out or _passthrough(
            user_text,
            "Gemini failed or returned invalid JSON; searching with your text as-is.",
            "none",
        )

    if prov in ("huggingface", "hf"):
        out = await _try_llm_json("huggingface", lambda: _huggingface_chat(user_text))
        return out or _passthrough(
            user_text,
            "Hugging Face chat failed or needs a free token (HUGGINGFACE_API_KEY). Searching with your text as-is.",
            "none",
        )

    if prov == "ollama":
        out = await _try_llm_json("ollama", lambda: _ollama_chat(user_text))
        return out or _passthrough(
            user_text,
            "Ollama unavailable or invalid response; searching with your text as-is.",
            "none",
        )

    if prov == "openai_compatible":
        if not settings.openai_compat_base_url or not settings.openai_compat_api_key:
            return _passthrough(
                user_text,
                "OpenAI-compatible API not fully configured; searching with your text as-is.",
                "none",
            )
        out = await _try_llm_json("openai_compatible", lambda: _openai_compatible_chat(user_text))
        return out or _passthrough(
            user_text,
            "OpenAI-compatible LLM failed; searching with your text as-is.",
            "none",
        )

    return _passthrough(
        user_text,
        f"Unknown LLM_PROVIDER={settings.llm_provider!r}; searching with your text as-is.",
        "none",
    )


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
        raise ValueError("GEMINI_API_KEY empty")

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


async def _huggingface_chat(user_text: str) -> str:
    """OpenAI-compatible chat on Hugging Face Router (optional HF token)."""
    base = settings.huggingface_router_url.rstrip("/")
    url = f"{base}/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    tok = settings.huggingface_api_key.strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    model = (settings.huggingface_model or "meta-llama/Llama-3.2-1B-Instruct").strip()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"raw": r.text[:500]}
            raise ValueError(f"Hugging Face HTTP {r.status_code}: {err}")
        body = r.json()

    choices = body.get("choices") or []
    if not choices:
        raise ValueError(f"Hugging Face returned no choices: {body}")
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "").strip()


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
