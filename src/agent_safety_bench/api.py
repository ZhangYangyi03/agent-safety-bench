"""API client: Universal OpenAI-compatible model caller with retry logic.

Supports any OpenAI-compatible API (AIPING, local Ollama, OpenAI, etc.)
with automatic retries, error handling, and proxy support.
"""

import os
import json
import time
from typing import Optional


def call_model(
    model: str,
    messages: list[dict],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    proxies: Optional[dict] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 3,
    timeout: int = 120,
) -> str:
    """Call an OpenAI-compatible chat API with retry logic.

    Args:
        model: Model name (e.g. "qwen3-8b", "gpt-4o")
        messages: Message list [{"role": "...", "content": "..."}]
        api_base: API base URL. Defaults to AIPING_ENDPOINT env or
                  "https://aiping.cn/api/v1"
        api_key: API key. Defaults to AIPING_API_KEY env.
        proxies: Proxy dict for requests (e.g. {"http": "...", "https": "..."})
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        max_retries: Number of retries on failure
        timeout: Request timeout in seconds

    Returns:
        Response text, or "ERROR: <msg>" on failure
    """
    import requests

    # Resolve defaults
    if api_base is None:
        api_base = os.environ.get(
            "AIPING_ENDPOINT",
            "https://aiping.cn/api/v1"
        )
    if api_key is None:
        api_key = os.environ.get("AIPING_API_KEY", "")

    url = f"{api_base.rstrip('/')}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                proxies=proxies,
                timeout=timeout,
            )
            data = resp.json()

            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                return content

            error_msg = data.get("error", {}).get("message", str(data))
            return f"ERROR: {error_msg}"

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            return f"ERROR: Timeout after {max_retries} attempts"

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            return f"ERROR: {e}"

    return "ERROR: Max retries exceeded"


def call_ollama(
    model: str,
    prompt: str,
    ollama_url: str = "http://localhost:11434/api/generate",
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> str:
    """Call a local Ollama model directly.

    Args:
        model: Model name (e.g. "qwen2.5:1.5b")
        prompt: Raw prompt text
        ollama_url: Ollama API URL
        temperature: Sampling temperature
        max_tokens: Max tokens

    Returns:
        Response text, or "ERROR: <msg>" on failure
    """
    import requests

    try:
        resp = requests.post(
            ollama_url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            },
            timeout=120,
        )
        return resp.json().get("response", "").strip()

    except Exception as e:
        return f"ERROR: {e}"


def list_available_models(api_base: Optional[str] = None,
                           api_key: Optional[str] = None,
                           proxies: Optional[dict] = None) -> list[dict]:
    """List available models from an OpenAI-compatible API.

    Returns:
        List of model info dicts
    """
    import requests

    if api_base is None:
        api_base = os.environ.get("AIPING_ENDPOINT", "https://aiping.cn/api/v1")
    if api_key is None:
        api_key = os.environ.get("AIPING_API_KEY", "")

    url = f"{api_base.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        resp = requests.get(url, headers=headers, proxies=proxies, timeout=30)
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        return [{"error": str(e)}]