import csv
import statistics
import glob


def analyze_file(filename):

    similarities = []

    with open(filename, encoding="utf-8") as file:

        reader = csv.reader(file, delimiter=";")

        header = next(reader, None)

        similarity_index = None

        # найти колонку similarity
        if header:
            for i, col in enumerate(header):
                if "similarity" in col.lower():
                    similarity_index = i
                    break

        if similarity_index is None:
            print(f"Similarity column not found in {filename}")
            return filename, 0, 0, 0, 0

        for row in reader:

            try:
                similarity = float(row[similarity_index])
                similarities.append(similarity)

            except:
                continue

    if not similarities:
        return filename, 0, 0, 0, 0

    avg = statistics.mean(similarities)
    max_val = max(similarities)
    min_val = min(similarities)

    return filename, avg, max_val, min_val, len(similarities)


def main():

    files = glob.glob("embedding_benchmark_*.csv")

    if not files:
        print("No CSV files found")
        return

    results = []

    for file in files:
        results.append(analyze_file(file))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\nEmbedding comparison:\n")

    for filename, avg, max_val, min_val, count in results:

        print(f"File: {filename}")
        print(f"Entries: {count}")
        print(f"Average similarity: {avg:.4f}")
        print(f"Max similarity: {max_val:.4f}")
        print(f"Min similarity: {min_val:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()