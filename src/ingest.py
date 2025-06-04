# src/ingest.py

import os
import json
import asyncio
import logging
import yaml

import aiohttp
from readability import Document
from bs4 import BeautifulSoup, Doctype, NavigableString, Tag
from transformers import AutoTokenizer

# ——— Config loader and helpers —————————————————————————————————
def load_config() -> dict:
    src = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(src)
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def sanitize(url: str) -> str:
    return ''.join(c if c.isalnum() or c == '-' else '_' for c in url)

def detect_lang_from_soup(soup: BeautifulSoup) -> str:
    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang", "").strip():
        return html_tag["lang"].strip()
    text = soup.get_text(" ", strip=True)
    from langdetect import detect, LangDetectException
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def extract_markdown(soup: BeautifulSoup) -> str:
    SKIP = ["nav", "footer", "aside", "script", "style"]
    for tag in soup.find_all(SKIP):
        tag.decompose()

    def _rec(t: Tag) -> str:
        parts = []
        for c in t.children:
            if isinstance(c, Doctype):
                continue
            if isinstance(c, NavigableString):
                parts.append(str(c))
            elif isinstance(c, Tag):
                n = c.name.lower()
                if n in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    lvl = int(n[1])
                    parts.append(f"\n{'#'*lvl} {c.get_text(strip=True)}\n\n")
                elif n == "p":
                    parts.append(_rec(c) + "\n\n")
                elif n in ("strong", "b"):
                    parts.append(f"**{c.get_text(strip=True)}**")
                elif n in ("em", "i"):
                    parts.append(f"_{c.get_text(strip=True)}_")
                elif n == "a":
                    parts.append(f"[{c.get_text(strip=True)}]({c.get('href')})")
                elif n == "ul":
                    for li in c.find_all("li", recursive=False):
                        parts.append(f"- {_rec(li)}\n")
                elif n == "ol":
                    for i, li in enumerate(c.find_all("li", recursive=False), 1):
                        parts.append(f"{i}. {_rec(li)}\n")
                elif n == "img":
                    parts.append(f"![{c.get('alt','')}]({c.get('src')})")
                else:
                    parts.append(_rec(c))
        return "".join(parts)

    md = _rec(soup)
    # удаляем лишние пустые строки
    return "\n\n".join([b.strip() for b in md.splitlines() if b.strip()])

# ——— Main ——————————————————————————————————————————————————————
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest")

cfg = load_config()
ROOT = project_root()

RAW_HTML = os.path.join(ROOT, cfg["data"]["raw_html_dir"])
RAW_JSON = os.path.join(ROOT, cfg["data"]["raw_json_dir"])
CHUNKS   = os.path.join(ROOT, cfg["data"]["chunks_dir"])
os.makedirs(RAW_HTML, exist_ok=True)
os.makedirs(RAW_JSON, exist_ok=True)
os.makedirs(CHUNKS,   exist_ok=True)

URLS       = cfg["urls"]
MAX_WORK   = cfg["ingest"]["max_fetch_workers"]
TIMEOUT    = cfg["ingest"]["request_timeout"]
CHUNK_SZ   = cfg["ingest"]["chunk_size"]
OVERLAP    = cfg["ingest"]["chunk_overlap"]

tokenizer = AutoTokenizer.from_pretrained(cfg["embed"]["model_name"])
HEADERS   = {"User-Agent":"Mozilla/5.0"}

def chunk_text(text: str) -> list[dict]:
    ids   = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    step  = CHUNK_SZ - OVERLAP
    for i in range(0, len(ids), step):
        win = ids[i:i+CHUNK_SZ]
        txt = tokenizer.decode(win, clean_up_tokenization_spaces=True)
        chunks.append({
            "chunk_id":    len(chunks),
            "text":        txt,
            "start_token": i,
            "end_token":   min(i+CHUNK_SZ, len(ids))
        })
        if i + CHUNK_SZ >= len(ids):
            break
    return chunks

async def fetch_parse(session, url, sem):
    async with sem:
        try:
            async with session.get(url, headers=HEADERS) as resp:
                resp.raise_for_status()
                html = await resp.text()
        except Exception as e:
            logger.warning(f"[FETCH ERR] {url}: {e}")
            return

    fn = sanitize(url)
    # 1) Сохраняем raw HTML
    with open(os.path.join(RAW_HTML, f"{fn}.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # 2) Readability → основной HTML + текст
    doc       = Document(html)
    main_html = doc.summary()
    main_txt  = BeautifulSoup(main_html, "html.parser").get_text(" ", strip=True)

    # 3) Язык и заголовок
    soup      = BeautifulSoup(html, "html.parser")
    lang      = detect_lang_from_soup(soup)
    title     = soup.title.string.strip() if soup.title else ""

    # 4) Markdown-альтернатива
    md        = extract_markdown(soup)

    # 5) Чанки
    chunks    = chunk_text(main_txt)

    # 6) Сохраняем JSON страницы
    page = {
        "page_url":   url,
        "page_title": title,
        "page_lang":  lang,
        "main_html":  main_html,
        "main_text":  main_txt,
        "md_text":    md,
        "chunks":     chunks
    }
    with open(os.path.join(RAW_JSON, f"{fn}.json"), "w", encoding="utf-8") as f:
        json.dump(page, f, ensure_ascii=False, indent=2)

    # 7) Сохраняем по отдельности каждый чан
    for ch in chunks:
        out = {
            "page_url":   url,
            "page_title": title,
            "page_lang":  lang,
            "chunk_id":   ch["chunk_id"],
            "text":       ch["text"]
        }
        with open(os.path.join(CHUNKS, f"{fn}_chunk_{ch['chunk_id']}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(f"[PARSED] {url}: {len(chunks)} chunks")

async def main():
    sem     = asyncio.Semaphore(MAX_WORK)
    to      = aiohttp.ClientTimeout(total=TIMEOUT)
    async with aiohttp.ClientSession(timeout=to) as s:
        await asyncio.gather(*(fetch_parse(s, u, sem) for u in URLS))

if __name__ == "__main__":
    asyncio.run(main())
