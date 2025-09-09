import re
import json
import os
from transformers import GPT2TokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer

# Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Heuristic regex patterns
HEURISTICS = {
    "roleplay": [
        r"\b(you are|you\'re|i am|i\'m|we are|we\'re)\b.+?\b(character|role|person|identity|name)\b",
        r"\b(what is your|tell me your|describe your)\s+(name|age|job|background|story)\?", 
        r"\b(pretend you are|let\'s pretend|in character|out of character)\b" 
    ],
    "rag": [
        r"\b(why|who|how|what|where)\b.+?\b(\?|answer|reason|explain|context)\b",
        r"\b(retrieve(d|s)?|retrieval|context|based on|according to)\b.+?\b(text|story|passage|information)\b", 
        r"\b(narrator|story|situation|scenario)\b.+?\b(ask|question|response|detail)\b"
    ],
    "reasoning": [
        r"\bif\b.+?\bthen\b",
        r"\bhow\s+many\b",
        r"\b(prove|deduce|derive|infer|therefore|thus|hence)\b" 
        
    ],
    "function_calling": [
        r"\b[\w\.]+\s*\(\s*\w+\s*=\s*[^,)]+(?:\s*,\s*\w+\s*=\s*[^,)]+)*\s*\)",
    ],
    
}

# TF-IDF vectorizers cache
tfidf_vectorizers = {}

def classify_text(text, task):
    """Regex heuristic classification."""
    for pattern in HEURISTICS[task]:
        if re.search(pattern, text, re.IGNORECASE):
            return 1
    return 0

def build_tfidf_model(examples, task):
    """Fit a TF-IDF model for given task."""
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([ex["text"] for ex in examples[:1000] if ex["text"]])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_vectorizers[task] = (vectorizer, feature_names[:10])
    return tfidf_vectorizers[task]

def tfidf_filter(text, vectorizer, feature_names, task_terms):
    """Check if text passes TF-IDF filter."""
    tfidf_score = sum(
        vectorizer.transform([text]).toarray()[0][i]
        for i, term in enumerate(feature_names) if term in task_terms
    )
    return tfidf_score > 0.1

def save_to_json(task, examples, token_count, cluster_id, output_dir):
    """Save filtered examples to JSON file."""
    instances = [{"text": ex["text"]} for ex in examples]
    if not instances:
        return
    filename = f"{task}_c{cluster_id}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump({"type": "text_only", "instances": instances}, f, indent=2)
    print(f"Saved {task} (cluster {cluster_id}): {len(instances)} instances, {token_count} tokens → {filepath}")
