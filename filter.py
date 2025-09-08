import os
from datasets import load_dataset
from tqdm import tqdm
from utils import (
    classify_text, build_tfidf_model, tfidf_filter,
    save_to_json, tokenizer, tfidf_vectorizers
)

# Config
max_examples_per_cluster = 50000
min_length, max_length = 256, 4096
task_clusters = {
    "reasoning": list(range(1, 8)),
    "function_calling": list(range(8, 15)),
    # "roleplay": list(range(15, 18)),
    # "rag": list(range(18, 21))
}
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def process_cluster(cluster_id, dataset):
    """Filter a single cluster by heuristics + TF-IDF."""
    task = next((t for t, clusters in task_clusters.items() if cluster_id in clusters), None)
    if not task:
        return [], 0

    filtered_examples, token_count_total = [], 0
    print(f"Processing cluster {cluster_id} (task: {task})")

    for example in dataset:
        text = example["text"]
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if min_length <= token_count <= max_length:
            if classify_text(text, task):
                filtered_examples.append({"text": text, "token_count": token_count})
                token_count_total += token_count
                if len(filtered_examples) >= max_examples_per_cluster:
                    break
        if len(filtered_examples) % 1000 == 0 and len(filtered_examples) > 0:
            print(f"  Processed {len(filtered_examples)} examples")

    # Apply TF-IDF refinement
    if filtered_examples:
        if task not in tfidf_vectorizers:
            build_tfidf_model(filtered_examples, task)
        vectorizer, feature_names = tfidf_vectorizers[task]
        task_terms = set(feature_names)
        filtered_examples = [
            ex for ex in filtered_examples if tfidf_filter(ex["text"], vectorizer, feature_names, task_terms)
        ]
        token_count_total = sum(ex["token_count"] for ex in filtered_examples)

    return filtered_examples, token_count_total

def main():
    dataset = load_dataset("OptimalScale/ClimbLab", split="train", streaming=True)

    token_counts = {task: 0 for task in task_clusters}
    example_counts = {task: 0 for task in task_clusters}

    for cluster_id in tqdm(range(1, 15), desc="Processing Clusters"):
        cluster_dataset = dataset.filter(lambda x: x.get("cluster_id", cluster_id) == cluster_id)
        filtered_examples, token_count = process_cluster(cluster_id, cluster_dataset)
        if filtered_examples:
            task = next(t for t, clusters in task_clusters.items() if cluster_id in clusters)
            save_to_json(task, filtered_examples, token_count, cluster_id, output_dir)
            token_counts[task] += token_count
            example_counts[task] += len(filtered_examples)

    print("\nFinal Summary:")
    print(f"Token Counts: {token_counts}")
    print(f"Example Counts: {example_counts}")
    print(f"Output Directory: {output_dir}")

if __name__ == "__main__":
    main()
