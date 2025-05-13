import requests

def generate_answer(prompt: str) -> str:
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False
        })
        return response.json().get("response", "[LLM не ответил]")
    except Exception as e:
        return f"[Ошибка]: {str(e)}"