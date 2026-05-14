# UNL Geoportal — natural language search

This project is a small **FastAPI** service plus a **static web UI** that helps people search your **ArcGIS Portal** catalog in plain language. Optional AI models rewrite a question into better Portal `q` keywords before calling the Portal [`/sharing/rest/search`](https://developers.arcgis.com/rest/users-groups-and-items/search/) API.

Repository: [wenjiewangwwj/unlgeoportalAI](https://github.com/wenjiewangwwj/unlgeoportalAI)

## What happens when someone searches

1. The user types a question (for example, “population near Lincoln”).
2. **Optional AI expansion** (first match wins):
   - **Bring your own key (browser → this API):** if the user pastes an **OpenAI** or **Anthropic (Claude)** API key in the UI, that provider is used to produce a short JSON plan (`portal_q`, optional `tags`, `user_note`).
   - **Otherwise (server-side):** if the host has `GEMINI_API_KEY` and/or a Hugging Face token (`HUGGINGFACE_API_KEY`, `HF_TOKEN`, or `HUGGINGFACEHUB_API_TOKEN`), those are tried according to `LLM_PROVIDER` (default **`auto`**). If nothing is configured, the app **still works**: it sends the user’s text to Portal search as-is (no 502 error).
3. The service builds a Portal search string. **Group scope:** when `PORTAL_GROUP_ID` is set on the server, the final `q` sent to Portal always includes `group:<id>` so results are limited to items shared with that group (for example the UNL Geoportal group `cdfaf0b822344c7792b688998094b1f0`). Clear `PORTAL_GROUP_ID` to search the whole portal.
4. The browser shows Portal item titles with links back to `.../home/item.html?id=...`.

## Repository layout

| Path | Purpose |
|------|--------|
| `backend/main.py` | FastAPI app, `POST /api/search`, `GET /health`, serves `frontend/index.html` at `/` |
| `backend/portal_search.py` | Calls Portal search; applies **group** scope |
| `backend/llm_expand.py` | LLM orchestration (BYOK + server Gemini / HF / …) |
| `backend/config.py` | Environment-driven settings (`pydantic-settings`) |
| `frontend/index.html` | Search UI (works on [GitHub Pages](https://pages.github.com/) or same host as the API) |
| `env.example` / `env.render` | Variable templates for local `.env` or [Render](https://render.com) |
| `render.yaml` | Optional Render **Web Service** blueprint |

## Deploy the API (Render)

1. Create a **Web Service** (not a Static Site) on Render and connect this GitHub repo.
2. **Start command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
3. **Build command:** `pip install -r requirements.txt`
4. Set environment variables (see `env.render`). Important keys:
   - **`PORTAL_SHARING_REST`** — Sharing REST root, e.g. `https://geoportal.nead.nebraska.edu/portal/sharing/rest`
   - **`PORTAL_GROUP_ID`** — optional; set to your Portal group GUID to restrict results
   - **`CORS_ORIGINS`** — include your GitHub Pages origin (e.g. `https://youruser.github.io`) if the UI is hosted there
   - **`LLM_PROVIDER=auto`** and optional **`GEMINI_API_KEY`** / **`HUGGINGFACE_API_KEY`** / **`HF_TOKEN`**
   - For Hugging Face natural-language expansion, set **one** of these on the server: `HUGGINGFACE_API_KEY`, `HF_TOKEN`, or `HUGGINGFACEHUB_API_TOKEN`
   - If you want the app to use Hugging Face without any browser-entered key, add the token only in the Render environment variables, not in the UI

## Host the UI (GitHub Pages or same origin)

- If the HTML is on **another origin** than the API, set `const API_BASE = "https://your-service.onrender.com"` in `frontend/index.html` (or use `?api=...`) and add that Pages origin to **`CORS_ORIGINS`** on the API.
- If you serve the UI from the **same** Render URL as the API, you can leave `API_BASE` empty.

## Bring your own API keys (OpenAI / Claude)

The UI has an optional section **“Bring your own model”**:

- **OpenAI:** API key + optional base URL (defaults to `https://api.openai.com/v1` for OpenAI, or use another OpenAI-compatible endpoint) + model (default `gpt-4o-mini`).
- **Claude (Anthropic):** API key + model (default `claude-3-5-haiku-20241022`).
- **Server-side Hugging Face:** no UI input is needed if the backend has a Hugging Face token configured. That is the default path for natural-language expansion when `LLM_PROVIDER=auto` and no Gemini key is set.

### Hugging Face deployment checklist

If you want natural-language search to work out of the box on Render:

1. Open your Render Web Service.
2. Go to **Environment**.
3. Add a Hugging Face token in **one** of these variables: `HUGGINGFACE_API_KEY`, `HF_TOKEN`, or `HUGGINGFACEHUB_API_TOKEN`.
4. Leave `LLM_PROVIDER=auto` unless you want to force a different backend.
5. Redeploy the service.

When this is set, users can type normal questions like “population near Lincoln” and the backend will try Hugging Face first, then fall back to plain Portal keyword search if the model call fails.

**Security note:** keys are sent to **this backend** over HTTPS on each search. They are **not** intended to be stored server-side by this app, but anyone operating the server could misconfigure logging. Do **not** use production secrets on a shared demo host you do not control. Prefer **server-side** keys in environment variables for production, or run your own deployment.

## Local development

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy env.example .env
# Edit .env — set PORTAL_SHARING_REST at minimum
python -m uvicorn backend.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/` and try **GET** `/health` to confirm Portal URL and group id.

## References

- [ArcGIS REST: Search (items, groups, users)](https://developers.arcgis.com/rest/users-groups-and-items/search/)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)
- [Hugging Face Inference Providers / Router](https://huggingface.co/docs/inference-providers/tasks/chat-completion)
- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat)
