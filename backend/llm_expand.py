import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from backend.config import settings

SYSTEM_PROMPT = """You help users find datasets in an ArcGIS Portal catalog.
Given a natural-language question, output a JSON object ONLY (no markdown) with:
- "portal_q": a short search string for ArcGIS Portal /sharing/rest/search parameter q.
-  Use relevant synonyms and broad dataset terms. Under 220 characters.
- "portal_q_variants": array of 0 to 4 alternate search strings, ordered from broadest to most specific.
- "tags": array of 0 to 3 broad tag-like keywords without "tags:" prefix, or empty array.
- "user_note": one short sentence explaining what you search for (for the UI).

Rules:
- Prefer broad keyword search terms over restrictive filters.
- Avoid hard filters like tags:... unless you are confident they help.
- Include common synonyms only when they broaden recall without becoming generic.
- If the user asks for a topic like farmland, wetlands, roads, hydrology, population, or land cover, infer natural domain synonyms yourself instead of relying on hardcoded code rules."""

HF_PUBLIC_MODELS = [
    "google/flan-t5-base",
    "microsoft/DialoGPT-medium",
    "facebook/blenderbot-400M-distill",
    "microsoft/DialoGPT-small",
    "gpt2",
]

HF_PUBLIC_PROMPT = """Rewrite the user's request into a short ArcGIS Portal search query.
Return only the search keywords, no explanation, no bullet points, and no markdown.
Prefer concise dataset/search terms and useful synonyms.

User request:
{user_text}

Search query:"""

_FILLER_PREFIXES = (
    "i am interested in ",
    "i'm interested in ",
    "i want ",
    "i need ",
    "i am looking for ",
    "i'm looking for ",
    "show me ",
    "find ",
    "please show me ",
    "please find ",
    "can you show me ",
    "can you find ",
    "tell me about ",
    "what is ",
    "information about ",
    "data about ",
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "around",
    "can",
    "data",
    "dataset",
    "datasets",
    "find",
    "for",
    "get",
    "i",
    "in",
    "interested",
    "is",
    "me",
    "need",
    "of",
    "on",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "want",
    "what",
    "you",
}


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("LLM did not return a JSON object")
    return json.loads(m.group())


def _passthrough(user_text: str, user_note: str, llm_mode: str = "none") -> dict[str, Any]:
    return {
        "portal_q": user_text,
        "portal_q_variants": [],
        "tags": [],
        "user_note": user_note,
        "llm_used": False,
        "llm_mode": llm_mode,
    }


def _basic_keyword_cleanup(user_text: str) -> str:
    text = user_text.strip()
    lowered = text.lower()
    for prefix in _FILLER_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            lowered = text.lower()
            break
    text = re.sub(r"^\s*(?:please|kindly)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text)
    if not words:
        return user_text.strip()

    kept = [w for w in words if w.lower() not in _STOPWORDS]
    if kept:
        cleaned = " ".join(kept).strip()
    else:
        cleaned = " ".join(words).strip()
    return cleaned


def cleanup_query_keywords(user_text: str) -> str:
    return _basic_keyword_cleanup(user_text)


def _coerce_portal_q(text: str, fallback: str) -> str:
    candidate = text.strip()
    if not candidate:
        return fallback
    candidate = re.sub(r"(?i)^(search query|query|keywords?|portal q)\s*[:\-]\s*", "", candidate).strip()
    candidate = candidate.splitlines()[0].strip(" `\"'.,;:-")
    candidate = re.sub(r"\s+", " ", candidate).strip()
    if not candidate:
        return fallback

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", candidate)
    if not words:
        return fallback

    stopword_ratio = sum(1 for w in words if w.lower() in _STOPWORDS) / max(len(words), 1)
    if len(words) > 12 or stopword_ratio > 0.55:
        return fallback
    return candidate[:220].strip()


def _normalize_llm_payload(data: dict[str, Any], llm_mode: str) -> dict[str, Any]:
    portal_q = str(data.get("portal_q", "")).strip()
    portal_q_variants = data.get("portal_q_variants") or []
    if not isinstance(portal_q_variants, list):
        portal_q_variants = []
    portal_q_variants = [str(t).strip() for t in portal_q_variants if str(t).strip()][:4]
    tags = data.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if str(t).strip()][:5]
    user_note = str(data.get("user_note", "")).strip()
    return {
        "portal_q": portal_q,
        "portal_q_variants": portal_q_variants,
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


async def _try_hf_public_models(user_text: str) -> dict[str, Any] | None:
    fallback_q = _basic_keyword_cleanup(user_text)
    prompt = HF_PUBLIC_PROMPT.format(user_text=user_text)
    token = settings.huggingface_api_key.strip()

    for model in HF_PUBLIC_MODELS:
        try:
            raw = await _huggingface_public_infer(model, prompt, token=token)
            portal_q = _coerce_portal_q(raw, fallback_q)
            if portal_q:
                return {
                    "portal_q": portal_q,
                    "portal_q_variants": [],
                    "tags": [],
                    "user_note": f"Hugging Face public model: {model}",
                    "llm_used": True,
                    "llm_mode": f"huggingface_public:{model}",
                }
        except Exception:
            continue

    if fallback_q and fallback_q.lower() != user_text.strip().lower():
        return {
            "portal_q": fallback_q,
            "portal_q_variants": [],
            "tags": [],
            "user_note": "Rule-based cleanup of your query; no LLM was available.",
            "llm_used": False,
            "llm_mode": "rule_based",
        }
    return None


async def expand_query_natural_language(
    user_text: str,
    *,
    user_openai_api_key: str = "",
    user_openai_base_url: str = "",
    user_openai_model: str = "",
    user_anthropic_api_key: str = "",
    user_anthropic_model: str = "",
) -> dict[str, Any]:
    """BYOK (OpenAI / Anthropic) takes priority over server-side LLM settings."""
    user_text = user_text.strip()
    if not user_text:
        return _passthrough("", "", "none")

    okey = user_openai_api_key.strip()
    obase = (user_openai_base_url or "https://api.openai.com/v1").strip().rstrip("/")
    omodel = (user_openai_model or "gpt-4o-mini").strip()
    akey = user_anthropic_api_key.strip()
    amodel = (user_anthropic_model or "claude-3-5-haiku-20241022").strip()

    if okey:
        out = await _try_llm_json(
            "openai_byok",
            lambda: _openai_bearer_chat(user_text, okey, obase, omodel),
        )
        if out:
            return out

    if akey:
        out = await _try_llm_json(
            "anthropic_byok",
            lambda: _anthropic_messages_chat(user_text, akey, amodel),
        )
        if out:
            return out

    if okey or akey:
        return _passthrough(
            user_text,
            "Bring-your-own-key request failed or returned invalid JSON; searching with your text as-is.",
            "none",
        )

    return await _expand_from_server_env(user_text)


async def _expand_from_server_env(user_text: str) -> dict[str, Any]:
    prov = settings.llm_provider.lower().strip()

    if prov in ("none", "off", "disabled"):
        cleaned = _basic_keyword_cleanup(user_text)
        if cleaned and cleaned != user_text.strip():
            return {
                "portal_q": cleaned,
                "portal_q_variants": [],
                "tags": [],
                "user_note": "LLM disabled (LLM_PROVIDER=none). Using a rule-based keyword cleanup.",
                "llm_used": False,
                "llm_mode": "rule_based",
            }
        return _passthrough(user_text, "LLM disabled (LLM_PROVIDER=none). Portal search uses your words as-is.", "none")

    if prov == "auto":
        out = await _try_hf_public_models(user_text)
        if out:
            return out
        if settings.gemini_api_key.strip():
            out = await _try_llm_json("gemini", lambda: _gemini_chat(user_text))
            if out:
                return out
        out = await _try_llm_json("huggingface", lambda: _huggingface_chat(user_text))
        if out:
            return out
        return _passthrough(
            _basic_keyword_cleanup(user_text),
            "No server LLM was available; using a cleaned keyword search string.",
            "rule_based",
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
        out = await _try_hf_public_models(user_text)
        if out:
            return out
        out = await _try_llm_json("huggingface", lambda: _huggingface_chat(user_text))
        return out or _passthrough(
            _basic_keyword_cleanup(user_text),
            "Hugging Face chat failed; using a cleaned keyword search string.",
            "rule_based",
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
                "OpenAI-compatible API not fully configured on server; searching with your text as-is.",
                "none",
            )
        out = await _try_llm_json("openai_compatible", lambda: _openai_compatible_chat(user_text))
        return out or _passthrough(
            user_text,
            "OpenAI-compatible LLM failed; searching with your text as-is.",
            "none",
        )

    return _passthrough(
        _basic_keyword_cleanup(user_text),
        f"Unknown LLM_PROVIDER={settings.llm_provider!r}; using a cleaned keyword search string.",
        "rule_based",
    )


async def _openai_bearer_chat(user_text: str, api_key: str, base_url: str, model: str) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.2,
    }
    if any(x in model.lower() for x in ("gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo", "o1", "o3")):
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"raw": r.text[:500]}
            raise ValueError(f"OpenAI HTTP {r.status_code}: {err}")
        body = r.json()

    choices = body.get("choices") or []
    if not choices:
        raise ValueError(f"OpenAI returned no choices: {body}")
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "").strip()


async def _anthropic_messages_chat(user_text: str, api_key: str, model: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_text}],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"raw": r.text[:500]}
            raise ValueError(f"Anthropic HTTP {r.status_code}: {err}")
        data = r.json()

    parts = data.get("content") or []
    texts: list[str] = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            t = p.get("text")
            if t:
                texts.append(str(t))
    return "\n".join(texts).strip()


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
    cparts = (candidates[0].get("content") or {}).get("parts") or []
    texts: list[str] = []
    for p in cparts:
        t = p.get("text")
        if t:
            texts.append(str(t))
    return "\n".join(texts).strip()


async def _huggingface_chat(user_text: str) -> str:
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


async def _huggingface_public_infer(model: str, prompt: str, token: str = "") -> str:
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 64,
            "return_full_text": False,
            "temperature": 0.2,
            "do_sample": True,
        },
        "options": {"wait_for_model": True},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"raw": r.text[:500]}
            raise ValueError(f"Hugging Face Inference API HTTP {r.status_code}: {err}")
        body = r.json()

    if isinstance(body, list):
        for item in body:
            if isinstance(item, dict):
                for key in ("generated_text", "summary_text", "text"):
                    value = item.get(key)
                    if value:
                        return str(value).strip()
    if isinstance(body, dict):
        if body.get("error"):
            raise ValueError(f"Hugging Face error: {body['error']}")
        for key in ("generated_text", "summary_text", "text"):
            value = body.get(key)
            if value:
                return str(value).strip()
    return ""
