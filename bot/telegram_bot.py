import sys
sys.path.append("C:\\Users\\nurbo\\Desktop\\coding\\python\\Uni\\Diploma\\virtual_assistant_rag")

import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from backend.rag_engine import answer_question

API_TOKEN = os.getenv("TELEGRAM_API_TOKEN") or "6719920104:AAFfMFnvGD_EAljDdtqvvxB3HWZiwD_nIwk"

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.reply(
        "Привет! Я виртуальный помощник AITU. Задай мне вопрос о поступлении, факультетах и т.д.\n"
    )

@dp.message()
async def handle_question(message: types.Message):
    text = message.text
    response = answer_question(text)
    await message.reply(response)

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
