from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    portal_sharing_rest: str
    portal_token: str = ""

    # gemini = Google AI Studio (free tier API key) | openai_compatible = Groq, etc. | ollama = local
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

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
