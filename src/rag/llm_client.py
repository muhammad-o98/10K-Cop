"""
LLM client factory. Defaults to local Ollama for free/local development.

Environment variables (optional):
  OLLAMA_BASE_URL: default http://localhost:11434
  OLLAMA_MODEL: e.g., llama3.1:8b-instruct
  OLLAMA_TEMPERATURE: float, default 0.2
"""

import os
from typing import Any

from langchain_community.llms import Ollama


def get_llm() -> Any:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct")
    temperature = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
    return Ollama(model=model, base_url=base_url, temperature=temperature)