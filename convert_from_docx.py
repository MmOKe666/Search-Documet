import os
from docx import Document

def docx_to_text(docx_path, text_folder):
    # Проверяем, существует ли выходная папка, если нет — создаем
    if not os.path.exists(text_folder):
        os.makedirs(text_folder)

    # Загружаем документ и извлекаем текст
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])

    # Генерируем путь для сохранения .txt
    file_name = os.path.basename(docx_path).replace('.docx', '.txt')
    output_path = os.path.join(text_folder, file_name)

    # Записываем текст в .txt файл
    with open(output_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

    print(f"Файл сохранён: {output_path}")

# Путь к папке с .docx файлами
docx_folder = "C:/Савелий Волкович/python/Documents"
text_folder = "C:/Савелий Волкович/python/prog/Files/text_files"

# Обход всех .docx файлов в папке
for filename in os.listdir(docx_folder):
    if filename.lower().endswith(".docx"):
        file_path = os.path.join(docx_folder, filename)
        docx_to_text(file_path, text_folder)

print(f"TXT-файлы сохранены в: {text_folder}")