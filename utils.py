import re
import json
import os
from transformers import GPT2TokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer

# Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Heuristic regex patterns
HEURISTICS = {
    "reasoning": [
        r"\b(if.+?then|if.+?,\s*then)\b",
        r"\bbecause\b.+?\b(so|therefore|thus|hence)\b",
        r"\b(this implies|it follows that|we can conclude|as a result)\b",
        r"\b(prove|deduce|derive|infer)\b",
        r"\b(logic(al)? reasoning|syllogism|premise|conclusion)\b",
        r"\b(necessary|sufficient)\s(condition|assumption)\b"
    ],
    "function_calling": [
        r"\b(def|lambda)\s+\w+\s*\(.*?\)\s*:",
        r"\w+\s*\(.*?\)",
        r"\b(return|yield)\b.*",
        r"\b(params?|arguments?|kwargs|args)\b",
        r"\b(method|function|callback|handler)\b",
        r"\b(API\s+call|invoke|execute)\b.*?\bfunction\b"
    ],
    "roleplay": [
        r"\b(hello|hi|hey|greetings)[.!]?\b",
        r"\b(i am|i'm|you are|you're)\b.*?\b(bot|assistant|human|person)\b",
        r"\b(let's|shall we)\b.*\b(play|pretend|imagine|go on an adventure)\b",
        r"\b(my name is|call me|you can be)\b",
        r"\b(how are you\??|what's up\??)\b",
        r"\b(in character|out of character|OOC)\b"
    ],
    "rag": [
        r"\b(retrieve(d|s)?|retrieval)\b.+?\b(document|passage|chunk|snippet|text)\b",
        r"\bfrom\b.+?\b(corpus|knowledge base|docs|database|index)\b",
        r"\bsearch\b.+?\b(for|query|through)\b",
        r"\b(question answering|QA system|RAG model|contextual answer)\b",
        r"\bknowledge\s(graph|base|source)\b",
        r"\bsemantic\ssearch\b"
    ]
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
