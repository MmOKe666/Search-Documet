# Use Python 3.12  (3.13 is currently not supported by weaviate client v4)

## Weaviate query scripts:
# Vector search (semantic search)
python scripts/search_weaviate_v4.py "datetime functions" --method vector

# Keyword search (BM25)
python scripts/search_weaviate_v4.py "datetime functions" --method keyword

# Default is vector search
python scripts/search_weaviate_v4.py "datetime functions"



Файлы:
1. convert_from_docx - coздание .txt файлов из файлов .docx
2. create_mapping.py - создание .json файла из .txt файлов
3. docker_compose.yml - файл, который используется для запуска сервиса Weaviate (векторной базы данных) в контейнере Docker.
4. embed_store - скрипт, который делает загрузку и векторизацию документов, а затем отправляет их в локальную векторную базу данных Weaviate.
5. read_weaviate - скрипт, который выводит что загрузилось в базу данных Weaviate
6. query_weaviate - скрипт для запроса. Пока что не работает.

Инструкция:
1. Через convert_from_docx создать .txt файлы.
2. Через create_mapping создать .json файл, в котором будут храниться все данные из .txt файлов
3. Для запуска базы данных Weaviate через Docker нужно:

3.1. Установить Docker Desktop с официального сайта: https://www.docker.com

3.2. Файл docker_compose.yml с параметрами для контейнера.

3.3. В консоли прописать: docker-compose up -d

4. Через embed_store.py загрузить данные в базу данных Weaviate.
5. С помощью read_weaviate можно посмотреть, что загрузилось в базу данных.
