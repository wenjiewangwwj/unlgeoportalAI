"""
AI-assisted ArcGIS Portal search API.

Set PORTAL_SHARING_REST to your Sharing REST root, e.g.
https://geoportal.nead.nebraska.edu/portal/sharing/rest
The app calls GET .../search?q=...&f=json on that root.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import settings
from backend.llm_expand import expand_query_natural_language
from backend.portal_search import merge_tags_into_q, portal_search


def item_html_id_prefix() -> str:
    sr = settings.sharing_rest.rstrip("/")
    if sr.endswith("/sharing/rest"):
        root = sr[: -len("/sharing/rest")].rstrip("/")
    else:
        root = sr.rstrip("/")
    return f"{root}/home/item.html?id="


app = FastAPI(title="Geoportal NL Search", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    num: int = Field(20, ge=1, le=100)
    start: int = Field(1, ge=1)


@app.get("/health")
async def health():
    prov = settings.llm_provider.lower().strip()
    extra: dict = {"llm_provider": settings.llm_provider}
    if prov == "gemini":
        extra["gemini_key_configured"] = bool(settings.gemini_api_key.strip())
    elif prov == "openai_compatible":
        extra["openai_compat_key_configured"] = bool(settings.openai_compat_api_key.strip())
    return {"ok": True, "portal": settings.sharing_rest, **extra}


@app.post("/api/search")
async def api_search(body: SearchRequest):
    try:
        expanded = await expand_query_natural_language(body.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e!s}") from e

    portal_q = merge_tags_into_q(expanded.get("portal_q") or "", expanded.get("tags") or [])
    if not portal_q:
        portal_q = body.query.strip()

    try:
        raw = await portal_search(portal_q, num=body.num, start=body.start)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Portal search error: {e!s}") from e

    if raw.get("error"):
        raise HTTPException(status_code=502, detail=str(raw.get("error")))

    return {
        "natural_language_query": body.query,
        "expanded": expanded,
        "portal_q_used": portal_q,
        "item_html_id_prefix": item_html_id_prefix(),
        "portal_response": raw,
    }


_frontend = Path(__file__).resolve().parent.parent / "frontend"
if _frontend.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_frontend), html=True), name="ui")


@app.get("/")
async def root():
    index = _frontend / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"message": "POST /api/search with JSON {query, num?, start?}. Serve frontend/ separately or open /ui/"}
