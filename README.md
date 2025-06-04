```markdown
# AITU Telegram RAG-Assistant

AI-консультант для абитуриентов Astana IT University, построенный по схеме Retrieval-Augmented Generation (RAG).
Проект парсит сайт АИТУ → разбивает на чанк-фрагменты → создаёт эмбеддинги → строит FAISS-индекс → отвечает на вопросы в Telegram через aiogram + Ollama.

## 📁 Структура проекта

```

telegram\_assistant/
├── .env                   # Переменные окружения (Telegram Token, Ollama URL, модель)
├── config.yaml            # Список URL, директории, параметры RAG
├── requirements.txt       # Зависимости Python
└── src/
├── ingest.py          # Парсинг сайта → raw\_html, raw\_json, chunks
├── embed.py           # Генерация эмбеддингов & metadata
├── indexer.py         # Строит FAISS-индекс
├── retriever.py       # Поиск релевантных чанков
├── model.py           # Клиент Ollama → /api/generate, стриминг, выбор модели
├── rag\_engine.py      # Сборка единого промта (с <context>), вызов модели, логирование
└── telegram\_bot.py    # aiogram-бот с прогревом модели и командой /setmodel

````

## ⚙️ Установка

1. **Клонировать репозиторий**
   ```bash
   git clone <URL-к-репозиторию>
   cd telegram_assistant
````

2. **Создать виртуальное окружение и активировать**

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Установить зависимости**

   ```bash
   pip install -r requirements.txt
   ```

4. **Создать файл `.env`** в корне проекта со следующим содержимым:

   ```dotenv
   TELEGRAM_TOKEN=<ваш_Telegram_Bot_Token>
   OLLAMA_URL=http://127.0.0.1:11434
   OLLAMA_MODEL=mistral
   ```

   * `TELEGRAM_TOKEN` — токен вашего бота Telegram.
   * `OLLAMA_URL` — адрес локального сервера Ollama (обычно `http://127.0.0.1:11434`).
   * `OLLAMA_MODEL` — модель по умолчанию (возможные: `mistral`, `deepseek`, `llama3`).

5. **Проверить конфигурацию**

   * Откройте `config.yaml` и убедитесь, что указаны нужные URL для парсинга и корректные директории.

## 🚀 Запуск проекта

### 1. Запустить сервер Ollama

В отдельном терминале:

```bash
ollama serve
```

Убедитесь в логах, что сервер слушает на `127.0.0.1:11434`.

### 2. Подготовить базу знаний

Выполните по порядку:

```bash
python -m src.ingest
```

* Скачивает и сохраняет raw HTML/JSON, разбивает страницы на чанки.

```bash
python -m src.embed
```

* Загружает чанки, генерирует эмбеддинги и сохраняет их (numpy + metadata).

```bash
python -m src.indexer
```

* Строит FAISS-индекс из эмбеддингов и сохраняет его вместе с метаданными.

### 3. Запустить Telegram-бота

```bash
python src/telegram_bot.py
```

* Бот автоматически «прогревает» (warm-up) модель Ollama одним коротким запросом.
* Затем запускает polling и ожидает входящих сообщений.

## 💬 Использование бота

1. **/start**
   Приветственное сообщение, показывает текущую модель.

2. **/setmodel `<deepseek|mistral|llama3>`**
   Смена модели на лету. По умолчанию в `.env` указано `mistral`.
   Пример:

   ```
   /setmodel deepseek
   ```

3. **Задайте любой вопрос об АИТУ**
   После обработки бот вернёт ответ (≤80 слов), основанный на найденных фрагментах.

   * Если информация найдена, выдаёт связный ответ с маркерами цитирования `[1]`, `[2]` и т. д.
   * Если ничего не нашлось, отвечает “Hmm, I’m not sure.”

## 📖 Пример работы

1. **Пользователь**:

   ```
   Какие документы нужно сдать?
   ```
2. **Бот**:

   ```
   • Копию удостоверения личности [1]
   • Медицинскую справку №075 с флюорографией [1]
   • Вакцинный сертификат №063 [1]
   • Фотографии 3×4 (6 шт.) [1]
   • Оригинал диплома или приложения [1]
   • Сертификат ЕНТ и чек об оплате (для платного обучения) [1]
   • Результаты теста AITU [1]
   ```

   Где `[1]` – номер фрагмента из `<context>`, в котором описаны все эти пункты.

## 🛠️ Возможные проблемы

* **Ollama не отвечает или 404**
  – Убедитесь, что `OLLAMA_URL` в `.env` совпадает с адресом из логов `Listening on 127.0.0.1:11434`.
  – Проверьте, что модель (mistral/deepseek/llama3) установлена в Ollama (`ollama list`).

* **Очень долгий первый ответ**
  – Во время старта Ollama загружает модель и прогревается. Прогрев выполняется в `telegram_bot.py`.
  – После прогрева ответы будут быстрее (модель уже загружена в память).

* **Пустые чанки / нет данных**
  – Проверьте, что все URL в `config.yaml` доступны и возвращают корректный HTML.
  – Убедитесь, что `src/ingest.py` завершился без ошибок, и в папках `raw_json` и `chunks` появились файлы.

## 📂 Итоговая структура директорий после первого прогрева

```
telegram_assistant/
├── data/
│   ├── raw_html/
│   ├── raw_json/
│   ├── chunks/
│   ├── embeddings/
│   │   ├── embeddings.npy
│   │   └── metadata.pkl
│   └── faiss_index/
│       ├── index.faiss
│       └── metadata.pkl
├── logs/
│   └── interactions.log
├── src/
│   ├── ingest.py
│   ├── embed.py
│   ├── indexer.py
│   ├── retriever.py
│   ├── model.py
│   ├── rag_engine.py
│   └── telegram_bot.py
├── config.yaml
├── .env
├── requirements.txt
└── README.md
```




