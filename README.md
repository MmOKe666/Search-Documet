# Search App 🤖

**Search App** — это AI-приложение для поиска информации по документам с использованием векторной базы данных Weaviate и LLM (OpenAI / OpenRouter / YandexGPT).

Приложение позволяет:

* конвертировать текстовые документы в JSON
* загружать документы в Weaviate Vector Database через Docker
* выполнять семантический поиск по документам
* общаться с AI-ассистентом через веб-интерфейс

---

# 🚀 Возможности

* 📄 Конвертация Markdown / TXT → JSON
* 🧠 Векторизация документов с использованием HuggingFace Transformers
* 🗄️ Хранение документов в Weaviate Vector Database
* 🔎 Семантический поиск по документам
* 🤖 AI-ассистент с использованием:

  * OpenRouter (OpenAI модели)
  * YandexGPT
* 🌐 Веб-интерфейс через Streamlit
* 🐳 Поддержка Docker

---

# 🏗️ Архитектура

Pipeline работы:

```
Text / Markdown Files
        ↓
text_to_json.py
        ↓
JSON files
        ↓
Docker (Weaviate)
        ↓
json_to_docker.py
        ↓
Weaviate Vector Database
        ↓
run_app.py
        ↓
Web Application (Streamlit)
```

---

# 📂 Структура проекта

```
Search-App/
│
├── text_to_json.py       # Конвертация документов в JSON
├── json_to_docker.py     # Загрузка JSON в Weaviate
├── run_app.py            # Запуск веб-приложения
├── app_weaviate.py       # Основное AI приложение
├── requirements.txt      # Зависимости Python
├── .env                  # API ключи и настройки
│
├── Files/
│   ├── text_files/      # Исходные документы
│   └── json/            # JSON файлы
```

---

# ⚙️ Требования

* Python 3.10+
* Docker
* Docker Compose
* API ключ одного из провайдеров:

  * OpenRouter API Key
    или
  * Yandex Cloud API Key + Folder ID

---

# 📦 Установка

## 1. Клонировать репозиторий

```
git clone https://github.com/yourusername/Search-App.git
cd Search-App
```

---

## 2. Установить зависимости

```
pip install -r requirements.txt
```

Основные библиотеки:

* streamlit
* weaviate-client
* langchain
* transformers
* torch
* sentence-transformers
* python-dotenv

---

## 3. Настроить .env файл

Пример:

```
WEAVIATE_URL=http://localhost:8080

OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=openai/gpt-3.5-turbo

YANDEX_API_KEY=your_key
YANDEX_FOLDER_ID=your_folder_id

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

# 🐳 Запуск Weaviate через Docker

В папке проекта:

```
docker-compose up -d
```

Проверка:

```
docker ps
```

---

# 📄 Шаг 1 — Конвертация документов

Помести файлы в:

```
Files/text_files/
```

Запусти:

```
python text_to_json.py
```

Результат:

```
Files/json/*.json
```

---

# 🗄️ Шаг 2 — Загрузка документов в Weaviate

```
python json_to_docker.py
```

Документы будут автоматически:

* векторизованы
* загружены в правильные коллекции

---

# 🤖 Шаг 3 — Запуск AI приложения

```
python run_app.py
```

Откроется браузер:

```
http://localhost:8501
```

---

# 💬 Использование

В веб-интерфейсе можно:

* выбрать LLM provider
* выбрать коллекции документов
* задать вопрос
* получить ответ на основе документов

---

# 🧠 Используемые технологии

* Python
* Streamlit
* Weaviate
* LangChain
* Transformers
* HuggingFace
* Docker
* OpenRouter API
* YandexGPT

---

# 🔎 Пример использования

Вопрос:

```
How does the component initialization work?
```

Ответ:

```
• Component initialization starts with configuration loading
• Required dependencies are injected
• System registers component in registry
```

---

# 🛠️ Troubleshooting

Если ошибка подключения к Weaviate:

```
docker-compose up -d
```

Если ошибка зависимостей:

```
pip install -r requirements.txt
```

---


