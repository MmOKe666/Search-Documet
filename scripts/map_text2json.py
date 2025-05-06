import json
import os
import uuid
import re

text_folder = os.path.normpath("../Files/text_files")
json_folder = os.path.normpath("../Files/json")
base_filename = "documentation_mapping"
extension = ".json"

documents = []

if not os.path.exists(json_folder):
    os.makedirs(json_folder)

for file_name in os.listdir(text_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(text_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Форматирование текста (заголовки в markdown)
            formatted_content = content

            # Главный заголовок
            formatted_content = re.sub(r'^\s*Спецификация\s*$', r'# Спецификация', formatted_content, flags=re.MULTILINE)

            # 1. Разделы верхнего уровня
            formatted_content = re.sub(r'^\s*(Обзор|Коды ошибок|Приложение А[^\n]*)',
                                       r'## \1', formatted_content, flags=re.MULTILINE)

            # 1. Главы: Компонент / Интерфейс
            formatted_content = re.sub(r'^\s*(Компонент\s+[^\n]+)', r'## \1', formatted_content, flags=re.MULTILINE)
            formatted_content = re.sub(r'^\s*(Интерфейс\s+[^\n]+)', r'## \1', formatted_content, flags=re.MULTILINE)

            # 1.1 Подглавы: Обзор
            formatted_content = re.sub(r'^\s*(Введение|Примечание|Ссылки)\s*$', r'## \1', formatted_content,flags=re.MULTILINE)

            # 1.1 Подглавы: описание интерфейсов
            formatted_content = re.sub(r'^\s*([^\n]+ описание на ECO IDL)', r'### \1', formatted_content,
                                       flags=re.MULTILINE)

            # 1.1.1 Функции
            formatted_content = re.sub(r'^\s*Функция\s+([^\n]+)', r'#### Функция \1', formatted_content,
                                       flags=re.MULTILINE)

            # Очистка лишних пустых строк
            formatted_content = re.sub(r'\n{3,}', '\n\n', formatted_content)

            content = formatted_content

            # Извлекаем метаданные
            title_match = re.search(r'Спецификация', content)
            version_match = re.search(r'Версия:\s*([^\n\r]+)', content)
            component_match = re.search(r'Компонент\s+([^\n\r]+)', content)
            overview_match = re.search(r'Обзор\s*([\s\S]+?)(?=\n\S|\Z)', content)
            author_match = re.search(r'Автор:\s*([^\n\r]+)', content)
            date_match = re.search(r'Дата:\s*([^\n\r]+)', content)

            metadata = {
                "title": "Спецификация" if title_match else "",
                "version": version_match.group(1).strip() if version_match else "",
                "component": component_match.group(1).strip() if component_match else "",
                "overview": overview_match.group(1).strip() if overview_match else "",
                "author": author_match.group(1).strip() if author_match else "",
                "date": date_match.group(1).strip() if date_match else ""
            }

            document = {
                "id": str(uuid.uuid4()),
                "content": content,
                "metadata": metadata
            }

            documents.append(document)

# Сохраняем в JSON
os.makedirs(json_folder, exist_ok=True)

# Initialize the filename
json_path = os.path.join(json_folder, base_filename + extension)

# Check if the file exists and increment the suffix if necessary
suffix = 1
while os.path.exists(json_path):
    # Create a new filename with the incremented suffix
    json_path = os.path.join(json_folder, f"{base_filename}_{suffix}{extension}")
    suffix += 1
    
# Now you can save your JSON file
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(documents, json_file, indent=4, ensure_ascii=False)

print(f"JSON-файл создан: {json_path}")