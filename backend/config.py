from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    portal_sharing_rest: str
    portal_token: str = ""
    # If set, all Portal /search queries are scoped to this group (ArcGIS q filter: group:<id>).
    # Leave empty to search the whole portal catalog.
    portal_group_id: str = "cdfaf0b822344c7792b688998094b1f0"

    # auto = try Gemini if key set, else Hugging Face router, else plain Portal search (no LLM).
    # none = never call an LLM. gemini | huggingface | ollama | openai_compatible = force that path (with fallback where noted).
    llm_provider: str = "auto"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Accept the common Hugging Face env names so deployment can use either one.
    huggingface_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("HUGGINGFACE_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
    )
    # A public instruction-tuned model works better for query expansion than a gated chat model.
    huggingface_model: str = "google/gemma-2-2b-it"
    huggingface_router_url: str = "https://router.huggingface.co/v1"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"

    openai_compat_base_url: str = ""
    openai_compat_api_key: str = ""
    openai_compat_model: str = ""

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://127.0.0.1:5500,http://localhost:5500"

    @property
    def sharing_rest(self) -> str:
        return self.portal_sharing_rest.rstrip("/")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
