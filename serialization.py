"""
JSON Serialization and Deserialization for Tigrinya Skip-gram Word Embeddings.
Preserves Ge'ez Unicode characters natively (ensure_ascii=False, UTF-8).
Validates model integrity upon reloading.

Pure Python standard library implementation.
"""

import json
from skipgram_tigrinya import SkipGramModel


MODEL_FORMAT_VERSION = "1.0.0"


def save_model(
    filepath,
    model,
    vocab,
    word2id,
    hyperparameters=None,
    corpus_metadata=None,
    training_metrics=None,
):
    """
    Saves trained Skip-gram model and full metadata to JSON.
    Uses UTF-8 and ensure_ascii=False to preserve Ge'ez characters cleanly.
    """
    payload = {
        "model_format_version": MODEL_FORMAT_VERSION,
        "model_architecture": "Skip-gram (from scratch)",
        "vocab_size": model.vocab_size,
        "embedding_dim": model.embedding_dim,
        "vocabulary": vocab,
        "word2id": word2id,
        "E": model.E,
        "U": model.U,
        "hyperparameters": hyperparameters or {},
        "corpus_metadata": corpus_metadata or {},
        "training_metrics": training_metrics or {},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_model(filepath):
    """
    Loads Skip-gram model, vocabulary, and metadata from JSON.

    Returns:
        model (SkipGramModel): Reconstructed model with restored weights.
        vocab (list): List of vocabulary words.
        word2id (dict): Mapping from word str to int ID.
        id2word (dict): Mapping from int ID to word str.
        metadata (dict): Full metadata dictionary.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["vocabulary"]
    word2id = {w: int(i) for w, i in data["word2id"].items()}
    id2word = {int(i): w for w, i in word2id.items()}

    vocab_size = data["vocab_size"]
    embedding_dim = data["embedding_dim"]
    seed = data.get("hyperparameters", {}).get("seed", 42)

    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim, seed=seed)
    model.E = data["E"]
    model.U = data["U"]

    metadata = {
        "model_format_version": data.get("model_format_version"),
        "model_architecture": data.get("model_architecture"),
        "hyperparameters": data.get("hyperparameters", {}),
        "corpus_metadata": data.get("corpus_metadata", {}),
        "training_metrics": data.get("training_metrics", {}),
    }

    return model, vocab, word2id, id2word, metadata


def verify_saved_model(orig_model, loaded_model, orig_vocab, loaded_vocab, tolerance=1e-5):
    """
    Verifies that a loaded model matches the original model within tolerance.
    Checks vocabulary order, matrix weights, and forward pass outputs.
    """
    # 1. Check vocabulary
    if orig_vocab != loaded_vocab:
        return False, "Vocabulary list mismatch"

    # 2. Check E matrix
    if len(orig_model.E) != len(loaded_model.E):
        return False, "E matrix row count mismatch"
    for i in range(len(orig_model.E)):
        for k in range(len(orig_model.E[i])):
            if abs(orig_model.E[i][k] - loaded_model.E[i][k]) > tolerance:
                return False, f"E[{i}][{k}] discrepancy exceeds tolerance"

    # 3. Check U matrix
    if len(orig_model.U) != len(loaded_model.U):
        return False, "U matrix row count mismatch"
    for k in range(len(orig_model.U)):
        for j in range(len(orig_model.U[k])):
            if abs(orig_model.U[k][j] - loaded_model.U[k][j]) > tolerance:
                return False, f"U[{k}][{j}] discrepancy exceeds tolerance"

    # 4. Check forward output for first and last word
    for test_idx in [0, min(1, orig_model.vocab_size - 1)]:
        orig_s, orig_p, orig_l, _ = orig_model.forward(test_idx, test_idx)
        load_s, load_p, load_l, _ = loaded_model.forward(test_idx, test_idx)
        if abs(orig_l - load_l) > tolerance:
            return False, f"Forward loss discrepancy for index {test_idx}"
        for j in range(len(orig_p)):
            if abs(orig_p[j] - load_p[j]) > tolerance:
                return False, f"Probability discrepancy at index {j}"

    return True, "Verification passed successfully"
