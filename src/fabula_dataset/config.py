from pydantic_settings import BaseSettings


class Config(BaseSettings):
    google_gemini_api_key: str = ""


config = Config()

