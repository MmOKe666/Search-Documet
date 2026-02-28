import json
import os
import uuid
import re
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_folder = os.path.normpath("../Files/text_files")
json_folder = os.path.normpath("../Files/json")
base_filename = "documentation_mapping_md"
extension = ".json"

documents = []

CHUNK_SIZE = 800
CHUNK_OVERLAP = 50


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n## ",
            "\n\n",
            "\n"
        ]
    )
    return splitter.split_text(text)


def split_front_matter(content):
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) == 3:
            yaml_part = parts[1]
            markdown_part = parts[2]
            return yaml.safe_load(yaml_part), markdown_part.strip()
    return {}, content

def clean_markdown(text: str) -> str:
    # Удаляем блоки кода ```cpp ... ```
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Удаляем inline код `
    text = re.sub(r"`", "", text)

    # Удаляем изображения
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"!image", "", text)

    # Удаляем spoiler блоки
    text = re.sub(r"<spoiler.*?>", "", text)
    text = re.sub(r"</spoiler>", "", text)

    # Удаляем HTML якоря
    text = re.sub(r"<a id=.*?>", "", text)
    text = re.sub(r"</a>", "", text)

    # Удаляем Markdown заголовки
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # Удаляем лишние ****
    text = text.replace("****", "")

    # Удаляем лишние **
    text = text.replace("**", "")

    # Убираем лишние пустые строки
    text = re.sub(r"\n\s*\n", "\n\n", text)

    # Удаляем markdown ссылки, оставляя текст
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Удаляем markdown таблицы
    text = re.sub(r"\|.*?\|\n", "", text)

    # Удаляем строку Table of Contents
    text = re.sub(r"Table of Contents.*?\n", "", text)



    return text.strip()

def detect_language(metadata, text):
    # 1. По source (RU.ECO / EN.ECO)
    source = metadata.get("source", "")
    if "RU.ECO" in source:
        return "RU"
    if "EN.ECO" in source:
        return "EN"

    # 2. По имени файла
    filename = metadata.get("fileName", "")
    if "RU." in filename:
        return "RU"
    if "EN." in filename:
        return "EN"

    # 3. По кириллице в тексте
    if re.search(r"[А-Яа-я]", text):
        return "RU"

    # 4. По умолчанию
    return "EN"


os.makedirs(json_folder, exist_ok=True)

for file_name in os.listdir(text_folder):
    if file_name.endswith(".md"):
        file_path = os.path.join(text_folder, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        metadata, markdown_content = split_front_matter(raw_content)

        markdown_content = clean_markdown(markdown_content)

        metadata["language"] = detect_language(metadata, markdown_content)

        print("Detected language:", metadata["language"])

        chunks = chunk_text(markdown_content)

        clean_title = metadata.get("title", "")
        if clean_title:
            clean_title = clean_title.replace("****", "")
            clean_title = clean_title.replace("**", "")
            clean_title = clean_title.strip()

        for i, chunk in enumerate(chunks):
            chunk_document = {
                "id": str(uuid.uuid4()),
                "chunk_id": f"{file_name}_chunk_{i + 1}",
                "type": metadata.get("documentType", "unknown"),
                "content": chunk,
                "metadata": {
                    "fileName": file_name,
                    "title": clean_title,
                    "component": metadata.get("componentName", ""),
                    "version": metadata.get("version", ""),
                    "CID": metadata.get("CID", ""),
                    "description": metadata.get("description", ""),
                    "tags": metadata.get("tags", ""),
                    "registryUrl": metadata.get("registryUrl", ""),
                    "source": metadata.get("source", ""),
                    "lastModified": metadata.get("lastModified", ""),
                    "language": metadata.get("language", "EN")
                }
            }

            documents.append(chunk_document)


json_path = os.path.join(json_folder, base_filename + extension)
suffix = 1

while os.path.exists(json_path):
    json_path = os.path.join(json_folder, f"{base_filename}_{suffix}{extension}")
    suffix += 1

with open(json_path, "w", encoding="utf-8") as json_file:
    json.dump(documents, json_file, indent=4, ensure_ascii=False)

print(f"JSON-файл создан: {json_path}")