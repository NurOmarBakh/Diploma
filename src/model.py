# src/model.py

import os
import json
import requests
from dotenv import load_dotenv

# грузим .env из корня проекта
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Глобальная переменная для текущей модели
CURRENT_MODEL = DEFAULT_MODEL

def set_model(name: str) -> None:
    """
    Сменить модель на лету: deepseek, mistral или llama3
    """
    global CURRENT_MODEL
    if name not in ("deepseek", "mistral", "llama3"):
        raise ValueError(f"Unsupported model '{name}'. Valid: deepseek, mistral, llama3.")
    CURRENT_MODEL = name

def generate_answer(prompt: str) -> str:
    """
    Отправляем prompt в Ollama через /api/generate с потоковой выдачей
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model":  CURRENT_MODEL,
        "prompt": prompt,
        "stream": True,
        "temperature": 0.1,
        # "max_tokens": 512
    }
    resp = requests.post(url, json=payload, stream=True, timeout=60)
    resp.raise_for_status()

    answer = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
            part = data.get("response", "")
            answer += part
        except json.JSONDecodeError:
            continue

    return answer.strip()
