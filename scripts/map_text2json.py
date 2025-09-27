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

# Chunking Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

for file_name in os.listdir(text_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(text_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Text formatting (headings in markdown)
            formatted_content = content

            # Main headline
            formatted_content = re.sub(r'^\s*Specification\s*$', r'# Specification', formatted_content, flags=re.MULTILINE)

            # 1. Top-level sections
            formatted_content = re.sub(r'^\s*(Overview|Error codes|Appendix A[^\n]*)',
                                       r'## \1', formatted_content, flags=re.MULTILINE)

            # 1. Chapters: Component / Interface
            formatted_content = re.sub(r'^\s*(Component\s+[^\n]+)', r'## \1', formatted_content, flags=re.MULTILINE)
            formatted_content = re.sub(r'^\s*(Interface\s+[^\n]+)', r'## \1', formatted_content, flags=re.MULTILINE)

            # 1.1 Sub-chapters: Overview
            formatted_content = re.sub(r'^\s*(Introduction|Note|Links)\s*$', r'## \1', formatted_content,flags=re.MULTILINE)

            # 1.1 Sub-chapters: description of interfaces
            formatted_content = re.sub(r'^\s*([^\n]+ IDL)', r'### \1', formatted_content,
                                       flags=re.MULTILINE)

            # 1.1.1 Functions
            formatted_content = re.sub(r'^\s*function\s+([^\n]+)', r'#### function \1', formatted_content,
                                       flags=re.MULTILINE)

            # Clearing unnecessary empty lines
            formatted_content = re.sub(r'\n{3,}', '\n\n', formatted_content)

            content = formatted_content

            # Extracting metadata
            title_match = re.search(r'Specification', content)
            version_match = re.search(r'Version:\s*([^\n\r]+)', content)
            component_match = re.search(r'Component\s+([^\n\r]+)', content)
            overview_match = re.search(r'Overview\s*([\s\S]+?)(?=\n\S|\Z)', content)
            author_match = re.search(r'Author:\s*([^\n\r]+)', content)
            date_match = re.search(r'Date:\s*([^\n\r]+)', content)
            status_match = re.search(r'Status:\s*([^\n\r]+)', content)
            type_match = re.search(r'\b(specification|documentation|user guide|example|)\b', content, re.IGNORECASE)

            doc_type = type_match.group(1).lower() if type_match else "unknown"
            component = component_match.group(1).strip() if component_match else ""

            metadata = {
                "fileName": file_name,
                "title": f"Specification of the {component} " if title_match and component else "",
                "version": version_match.group(1).strip() if version_match else "",
                "component": component_match.group(1).strip() if component_match else "",
                "overview": overview_match.group(1).strip() if overview_match else "",
                "author": author_match.group(1).strip() if author_match else "PeerF LLC",
                "date": date_match.group(1).strip() if date_match else "",
                "status": status_match.group(1).strip() if status_match else ""
            }

            # Splitting the content into chunks
            chunks = chunk_text(content)

            for i, chunk in enumerate(chunks):
                chunk_document = {
                    "id": str(uuid.uuid4()),
                    "chunk_id": f"{file_name}_chunk_{i + 1}",
                    "type": doc_type,
                    "content": chunk,
                    "metadata": metadata
                }
                documents.append(chunk_document)

# Saving it in JSON
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