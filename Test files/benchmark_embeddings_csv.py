import os
import csv
import datetime
from dotenv import load_dotenv

import weaviate
from weaviate.connect import ConnectionParams

from sentence_transformers import SentenceTransformer


# =========================
# Load environment variables
# =========================

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
)

COLLECTION_NAMES = [
    "Component_RU",
    "Component_EN",
    "Specification_RU",
    "Specification_EN",
    "Guide_RU",
    "Guide_EN"
]


# =========================
# Test questions
# =========================

TEST_QUESTIONS = [

    # Основные вопросы
    "What is Eco.List1 component?",
    "What does Add function do in Eco.List1?",
    "What is tree data structure?",
    "What does get_LeastCommonAncestor function do?",
    "Что такое структура данных дерево?",
    
    # Общие вопросы о компонентах
    "What is Eco.Tree1 component?",
    "What is Eco.List1 component?",
    "What does Eco.Tree1 implement?",
    "What does Eco.List1 implement?",

    # Tree structure questions
    "What is a tree data structure?",
    "What is a tree node?",
    "What is a root node in a tree?",
    "What is a leaf node?",
    "What is the difference between root and leaf node?",
    "What is depth in a tree?",
    "What is height in a tree?",

    # Tree API questions
    "What does CreateNode function do?",
    "What does InsertNode function do?",
    "What does DeleteNode function do?",
    "What does Clear function do in Eco.Tree1?",

    # Tree node interface questions
    "What does get_Parent function return?",
    "What does AddChild function do?",
    "How to add child node in Eco.Tree1Node?",

    # List questions
    "What is Eco.List1?",
    "What is list data structure?",
    "What does Count function do in Eco.List1?",
    "What does Remove function do?",
    "What does RemoveAt function do?",
    "What does InsertAt function do?",
    "What does IndexOf function do?",
    "What does Clear function do in Eco.List1?",

    # Practical usage questions
    "How to add element to list?",
    "How to remove element from list?",
    "How to get element by index?",
    "How to clear a list?",

    # Russian questions (important for multilingual testing)
    "Что такое компонент Eco.Tree1?",
    "Что такое компонент Eco.List1?",
    "Что такое узел дерева?",
    "Что делает функция Add?",
    "Что делает функция Remove?",
    "Что делает функция Clear?",
    "Как добавить элемент в список?",
    "Как удалить элемент из списка?",
]

# =========================
# CSV setup
# =========================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

CSV_FILENAME = f"embedding_benchmark_{timestamp}.csv"


# =========================
# Connect to Weaviate v4
# =========================

print(f"Connecting to Weaviate: {WEAVIATE_URL}")

client = weaviate.WeaviateClient(
    connection_params=ConnectionParams.from_url(
        url=WEAVIATE_URL,
        grpc_port=50051
    )
)

client.connect()


# =========================
# Load embedding model
# =========================

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)


# =========================
# Search function
# =========================

def search_all_collections(question):

    query_vector = model.encode(question).tolist()

    all_results = []

    for collection_name in COLLECTION_NAMES:

        collection = client.collections.get(collection_name)

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=3,
            return_metadata=["distance"]
        )

        for obj in response.objects:

            distance = obj.metadata.distance
            similarity = 1 - distance

            content = obj.properties.get("content", "")

            all_results.append({
                "collection": collection_name,
                "similarity": similarity,
                "content": content
            })

    # сортируем по similarity
    all_results.sort(key=lambda x: x["similarity"], reverse=True)

    return all_results[:3]


# =========================
# Benchmark
# =========================

def run_benchmark():

    similarities = []

    print("\nStarting benchmark...\n")

    with open(CSV_FILENAME, mode="w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file, delimiter=";")

        writer.writerow([
            "model",
            "question",
            "rank",
            "collection",
            "similarity",
            "preview"
        ])

        for question in TEST_QUESTIONS:

            print(f"\nQuestion: {question}")

            results = search_all_collections(question)

            if not results:
                print("No results found")
                continue

            for rank, result in enumerate(results, start=1):
                similarity = result["similarity"]
                content = result["content"]
                collection = result["collection"]

                similarities.append(similarity)

                preview = content[:100].replace("\n", " ")

                print(f"Rank {rank} | Collection: {collection} | Similarity: {similarity:.4f}")

                writer.writerow([
                    MODEL_NAME,
                    question,
                    rank,
                    collection,
                    similarity,
                    preview
                ])

    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
    else:
        avg_similarity = 0

    print("\n========================")
    print("Benchmark complete")
    print(f"Model: {MODEL_NAME}")
    print(f"Average similarity: {avg_similarity:.4f}")
    print(f"CSV saved to: {CSV_FILENAME}")
    print("========================\n")


# =========================
# Run
# =========================

if __name__ == "__main__":

    run_benchmark()

    client.close()