import os
import re
import pickle
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup, Doctype, NavigableString, Tag
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def langchain_docs_extractor(soup: BeautifulSoup) -> str:
    SCAPE_TAGS = ["nav", "footer", "aside", "script", "style"]
    for tag in soup.find_all(SCAPE_TAGS):
        tag.decompose()

    def get_text(tag: Tag) -> str:
        texts = []
        for child in tag.children:
            if isinstance(child, Doctype):
                continue
            if isinstance(child, NavigableString):
                texts.append(child)
            elif isinstance(child, Tag):
                if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    texts.append(f"{'#' * int(child.name[1:])} {child.get_text()}\n\n")
                elif child.name == "a":
                    texts.append(f"[{child.get_text(strip=False)}]({child.get('href')})")
                elif child.name == "img":
                    texts.append(f"![{child.get('alt', '')}]({child.get('src')})")
                elif child.name in ["strong", "b"]:
                    texts.append(f"**{child.get_text(strip=False)}**")
                elif child.name in ["em", "i"]:
                    texts.append(f"_{child.get_text(strip=False)}_")
                elif child.name == "br":
                    texts.append("\n")
                elif child.name == "code":
                    parent = child.find_parent()
                    if parent is not None and parent.name == "pre":
                        classes = parent.attrs.get("class", [])
                        language = ""
                        for cl in classes:
                            if re.match(r"language-\w+", cl):
                                language = cl.split("-")[1]
                                break
                        lines = []
                        for span in child.find_all("span", class_="token-line"):
                            line_content = "".join(token.get_text() for token in span.find_all("span"))
                            lines.append(line_content)
                        code_content = "\n".join(lines)
                        texts.append(f"```{language}\n{code_content}\n```\n\n")
                    else:
                        texts.append(f"`{child.get_text(strip=False)}`")
                elif child.name == "p":
                    texts.append(get_text(child))
                    texts.append("\n\n")
                elif child.name == "ul":
                    for li in child.find_all("li", recursive=False):
                        texts.append("- " + get_text(li) + "\n\n")
                elif child.name == "ol":
                    for i, li in enumerate(child.find_all("li", recursive=False)):
                        texts.append(f"{i + 1}. " + get_text(li) + "\n\n")
                elif child.name == "div" and "tabs-container" in child.get("class", []):
                    tabs = child.find_all("li", {"role": "tab"})
                    tab_panels = child.find_all("div", {"role": "tabpanel"})
                    for tab, tab_panel in zip(tabs, tab_panels):
                        tab_name = tab.get_text(strip=True)
                        texts.append(f"{tab_name}\n")
                        texts.append(get_text(tab_panel))
                elif child.name == "table":
                    thead = child.find("thead")
                    if thead:
                        headers = thead.find_all("th")
                        if headers:
                            texts.append("| " + " | ".join(header.get_text() for header in headers) + " |\n")
                            texts.append("| " + " | ".join("----" for _ in headers) + " |\n")
                    tbody = child.find("tbody")
                    if tbody:
                        for row in tbody.find_all("tr"):
                            texts.append("| " + " | ".join(cell.get_text(strip=True) for cell in row.find_all("td")) + " |\n")
                    texts.append("\n\n")
                elif child.name in ["button"]:
                    continue
                else:
                    texts.append(get_text(child))
        return "".join(texts)

    joined = get_text(soup)
    return re.sub(r"\n\n+", "\n\n", joined).strip()

def fetch_and_parse(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text = langchain_docs_extractor(soup)
            logger.info(f"Успешно извлечён текст с {url} (длина: {len(text)} символов)")
            return text
        else:
            logger.warning(f"Ошибка {response.status_code} при получении {url}")
            return ""
    except Exception as e:
        logger.exception(f"Ошибка при получении {url}: {e}")
        return ""

def build_knowledge_base(urls: list) -> list:
    texts = []
    for url in urls:
        logger.info(f"Загружаем: {url}")
        text = fetch_and_parse(url)
        if text:
            texts.append(text)
    return texts

def main():
    urls = [
        "https://admission.astanait.edu.kz/",
        "https://admission.astanait.edu.kz/aet",

        "https://astanait.edu.kz/programs/",
        "https://astanait.edu.kz/computer-science/",
        "https://astanait.edu.kz/software-engineering/",
        "https://astanait.edu.kz/big-data-analysis/",
        "https://astanait.edu.kz/media-technologies/",
        "https://astanait.edu.kz/mathematical-computational-science-3/",
        "https://astanait.edu.kz/big-data-in-healthcare/",
        "https://astanait.edu.kz/cybersecurity/",
        "https://astanait.edu.kz/smart-technologies/",
        "https://astanait.edu.kz/industrial-internet-of-things/",
        "https://astanait.edu.kz/electronic-engineering/",
        "https://astanait.edu.kz/it-management/",
        "https://astanait.edu.kz/ai-business/",
        "https://astanait.edu.kz/it-entrepreneurship/",
        "https://astanait.edu.kz/digital-journalism/",

        "https://admission.astanait.edu.kz/magistratura",
        "https://astanait.edu.kz/doktorantura/",

        "https://astanait.edu.kz/university-life/student-housing/",
        "https://astanait.edu.kz/university-life/student-council/",
        "https://astanait.edu.kz/foundation-aitu/",
        "https://astanait.edu.kz/about/"

    ]

    raw_texts = build_knowledge_base(urls)

    split_texts = []
    for text in raw_texts:
        chunks = text.split("\n\n")
        split_texts.extend(chunks)

    logger.info(f"Получено {len(split_texts)} текстовых кусков для индексации.")

    if not split_texts:
        logger.error("Нет данных для индексации. Завершение.")
        return

    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = EMBEDDING_MODEL.encode(split_texts, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    os.makedirs("vector_db1", exist_ok=True)
    with open("vector_db1/embeddings.pkl", "wb") as f:
        pickle.dump(split_texts, f)
    faiss.write_index(index, "vector_db1/index.faiss")
    logger.info("[+] Индексация завершена")

if __name__ == "__main__":
    main()
