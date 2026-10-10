import time
from google import genai
from google.genai import types
from typing import Literal, TypeVar

from pydantic import BaseModel

from fabula_dataset.config import config

T = TypeVar("T", bound=BaseModel)


def call_llm(
    prompt: str,
    model: Literal["gemini"],
    response_schema: type[T] | None = None,
) -> T | str | None:
    match model:
        case "gemini":
            call_gemini(prompt, response_schema)


def call_gemini(prompt: str, response_schema: type[T] | None = None) -> T | str | None:
    client = genai.Client(api_key=config.google_gemini_api_key)
    base_delay = 5
    attempts = 5
    for attempt in range(attempts):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.2,
                ),
            )

            if not response.text:
                return None

            # print("Raw Response - Gemini")
            # print(response.text)

            if response_schema:
                return response_schema.model_validate_json(response.text)
            return response.text
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = (
                "429" in err_msg
                or "503" in err_msg
                or "resource exhausted" in err_msg
                or "rate limit" in err_msg
                or "unavailable" in err_msg
            )
            if not is_rate_limit or attempt == attempts - 1:
                raise
            wait = base_delay * (2 * attempt)
            print(
                f"Rate limit hit, retrying in {wait}s (attempt {attempt + 1}/{attempts})..."
            )
            time.sleep(wait)

    return None
