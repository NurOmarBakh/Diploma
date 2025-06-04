# src/telegram_bot.py

import os
import logging
import asyncio

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandObject

from rag_engine import answer_question
from model      import generate_answer, set_model, CURRENT_MODEL

# Загрузка .env
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN not set")

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("telegram_bot_aiogram")

bot = Bot(token=TOKEN)
dp  = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(msg: types.Message):
    text = (
        "Привет! Я виртуальный помощник AITU.\n"
        "Задай мне любой вопрос о поступлении, факультетах и т.д.\n\n"
        f"Текущая модель: *{CURRENT_MODEL}*\n"
        "Чтобы сменить модель, используй команду /setmodel `<deepseek|mistral|llama3>`.\n"
    )
    await msg.answer(text, parse_mode="Markdown")

@dp.message(Command("setmodel"))
async def cmd_setmodel(msg: types.Message, command: CommandObject):
    """
    Ожидаем: /setmodel <имя>
    Допустимые имена: deepseek, mistral, llama3
    """
    args  = command.args or ""
    choice = args.strip().lower()
    if choice not in ("deepseek", "mistral", "llama3"):
        await msg.answer("❗ Неверное название модели. Варианты: deepseek, mistral, llama3.")
        return

    try:
        set_model(choice)
        await msg.answer(f"✅ Модель успешно сменена на *{choice}*.", parse_mode="Markdown")
    except ValueError as e:
        await msg.answer(f"❗ Ошибка: {e}")

@dp.message()
async def handle_q(msg: types.Message):
    q = msg.text.strip()
    logger.info(f"Received question: {q!r}")
    try:
        a = answer_question(q)
    except Exception:
        logger.exception("Error in RAG engine")
        a = "⚠️ Извините, при обработке запроса произошла ошибка."
    await msg.answer(a)

async def main():
    # 1) Warmup: прогрев модели одним коротким запросом
    try:
        _ = generate_answer(" ")
        logger.info("Ollama model warmed up successfully")
    except Exception:
        logger.exception("Model warmup failed; first request may be slow")

    # 2) Старт polling
    logger.info("Starting aiogram bot polling…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
