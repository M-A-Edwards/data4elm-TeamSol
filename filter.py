import os
from datasets import load_dataset
from tqdm import tqdm
from utils import (
    classify_text, sentence_count, split_input_output,
    save_to_json, tokenizer, HEURISTICS, tfidf_vectorizers
)
from datetime import datetime

# Config
max_examples_per_cluster = 2
min_input_tokens, max_input_tokens = 5, 200
min_output_tokens, max_output_tokens = 1, 100
min_input_sents, max_input_sents = 1, 5
min_output_sents, max_output_sents = 1, 5
task_clusters = {
    "reasoning": list(range(1, 6)), # + list(range(11, 16)),
    "rag": list(range(6, 11)), # + list(range(16, 21)),
}
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def process_cluster(cluster_id, dataset):
    """Filter a single cluster by heuristics."""
    task = next((t for t, clusters in task_clusters.items() if cluster_id in clusters), None)
    if not task:
        return [], 0

    filtered_examples, token_count_total = [], 0
    print(f"Processing cluster {cluster_id} (task: {task})")

    for i, example in enumerate(dataset):
        text = example["text"]
        if classify_text(text, task) > 0:
            input_text, output_text = split_input_output(text, task)
            if not input_text or not output_text:
                continue

            input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))

            input_sents = sentence_count(input_text)
            output_sents = sentence_count(output_text)

            if (
                min_input_tokens <= input_tokens <= max_input_tokens and
                min_output_tokens <= output_tokens <= max_output_tokens and
                min_input_sents <= input_sents <= max_input_sents and
                min_output_sents <= output_sents <= max_output_sents
            ):
                filtered_examples.append({
                    "input": input_text,
                    "output": output_text,
                    "token_count": input_tokens + output_tokens
                })
                token_count_total += input_tokens + output_tokens

                if len(filtered_examples) >= max_examples_per_cluster:
                    break

        if len(filtered_examples) % 1000 == 0 and len(filtered_examples) > 0:
            print(f"  Processed {len(filtered_examples)} examples")

    return filtered_examples, token_count_total

def load_cluster_data(cluster_id):
    try:
        dataset = load_dataset(
            "OptimalScale/ClimbLab",
            data_files={"train": f"cluster_{cluster_id}/*.parquet"},
            split="train",
            streaming=True
        )
        print(f"[{datetime.now()}] Loaded dataset for cluster {cluster_id}")
        return dataset
    except Exception as e:
        print(f"[{datetime.now()}] Error loading dataset for cluster {cluster_id}: {e}")
        return []

def main():
    token_counts = {task: 0 for task in task_clusters}
    example_counts = {task: 0 for task in task_clusters}

    for cluster_id in tqdm(range(1, 11), desc="Processing Clusters"):
        dataset = load_cluster_data(cluster_id)
        if not dataset:
            print(f"[{datetime.now()}] Skipping cluster {cluster_id} (no data)")
            continue
        filtered_examples, token_count = process_cluster(cluster_id, dataset)
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
