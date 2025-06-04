import os
from datetime import datetime
from typing import List

from retriever import retrieve
from model     import generate_answer

def log_interaction(question: str, answer: str) -> None:
    """
    Логируем запрос и ответ в файл logs/interactions.log
    """
    os.makedirs("logs", exist_ok=True)
    with open("logs/interactions.log", "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}]\nQ: {question}\nA: {answer}\n\n")

def build_prompt(context_chunks: List[dict], question: str) -> str:
    """
    Формирует единый промт, включающий:
    1) инструкцию (объединённый промт),
    2) блок <context> с нумерованными фрагментами,
    3) сам вопрос.
    Каждый фрагмент нумеруется начиная с 1 в том порядке,
    в котором они передаются в context_chunks.
    """
    # Инструкция (объединённый промт)
    instruction = (
        "You are an expert consultant and problem-solver, tasked with answering any question about Astana IT University. "
        "Generate a comprehensive and informative answer of 80 words or less for the given question based solely on the provided search results (URL and content). "
        "You must only use information from those results. Use an unbiased, journalistic tone. Combine search results into a coherent answer without repeating text. "
        "Use bullet points for readability. Cite search results using [${number}] notation immediately after the sentence or paragraph that references them. "
        "If different results refer to different entities with the same name, write separate answers for each. "
        "If there is no relevant information in the context, just say “Hmm, I’m not sure.” Anything between the following `<context>` tags is retrieved from a knowledge bank, not part of the conversation:\n\n"
    )

    context_lines = ["<context>"]
    for idx, chunk in enumerate(context_chunks, start=1):
        # Нумеруем каждый фрагмент
        url   = chunk.get("page_url", "")
        text  = chunk.get("text", "").strip().replace("\n", " ")
        # Добавляем в виде: "1. URL: … Text: …"
        context_lines.append(f"{idx}. URL: {url}\n   Text: {text}\n")
    context_lines.append("</context>\n")

    # Собираем финальный промт
    prompt = (
        instruction
        + "\n".join(context_lines)
        + f"Question: {question}\nAnswer:"
    )
    return prompt

def answer_question(question: str) -> str:
    """
    1) Получает топ-K фрагментов из реликванта (retrieve),
    2) Собирает их в нумерованный список,
    3) Формирует единый промт (build_prompt),
    4) Отправляет его в модель (generate_answer),
    5) Логирует взаимодействие и возвращает ответ.
    """
    # Берём, например, первые 10 релевантных фрагментов
    top_k = 10
    retrieved = retrieve(question)  # возвращает list[dict], где ключ "text" и остальные метаданные
    if not retrieved:
        return "Hmm, I’m not sure."

    # Оставляем максимум top_k штук
    context_chunks = retrieved[:top_k]
    # Тексты в запросе нужны именно из поля "text", остальное — для цитирования
    ctx_prompt = build_prompt(context_chunks, question)

    # Получаем ответ от модели (стриминг внутри)
    answer = generate_answer(ctx_prompt)

    log_interaction(question, answer)
    return answer