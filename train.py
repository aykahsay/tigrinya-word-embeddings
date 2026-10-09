"""
Training Pipeline for Tigrinya Skip-gram Word Embeddings from Scratch.
Implements data loading, preprocessing, sentence-level train/validation split,
manual SGD training, fixed-weight epoch loss evaluation, nearest-neighbor queries,
and JSON model serialization.

Pure Python standard library implementation.
"""

import argparse
import os
import random
import time
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from corpus import (
    load_raw_corpus_tsv,
    preprocess_corpus_lines,
    split_sentences_train_val,
    build_vocabulary,
    generate_skipgram_pairs,
)
from skipgram_tigrinya import SkipGramModel
from evaluation import (
    finite_difference_gradient_check,
    get_nearest_neighbors,
)
from serialization import save_model, load_model, verify_saved_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Tigrinya Skip-gram Word Embeddings from Scratch."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=os.path.join("data", "raw", "kit5k.tsv"),
        help="Path to raw parallel TSV corpus.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=10,
        help="Dimensionality of word vectors (default: 10).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2,
        help="Symmetric context window size (default: 2).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="SGD learning rate (default: 0.05).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=3,
        help="Minimum word frequency threshold (default: 3).",
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=600,
        help="Number of raw sentences to load for pilot training (default: 600).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proportion of sentences held out for validation (default: 0.15).",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default=os.path.join("artifacts", "tigrinya_skipgram_model.json"),
        help="Destination path for trained model JSON.",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default=os.path.join("reports", "training_metrics.json"),
        help="Destination path for training metrics JSON.",
    )
    return parser.parse_args()


def run_experiment(args):
    print("=" * 65)
    print("Tigrinya Word Embeddings from Scratch — Training Pipeline")
    print("=" * 65)
    print(f"Corpus: {args.corpus_path}")
    print(f"Max Sentences: {args.max_sentences}")
    print(f"Embedding Dim: {args.embedding_dim}")
    print(f"Context Window: {args.window}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print(f"Min Count: {args.min_count}")
    print("-" * 65)

    # 1. Load raw corpus
    if not os.path.exists(args.corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {args.corpus_path}")

    raw_sentences = load_raw_corpus_tsv(args.corpus_path, tig_col_index=1, limit=args.max_sentences)
    print(f"[1/8] Loaded {len(raw_sentences)} raw Tigrinya sentences.")

    # 2. Preprocess and segment
    tokenized_sentences = preprocess_corpus_lines(raw_sentences)
    total_tokens = sum(len(s) for s in tokenized_sentences)
    print(f"[2/8] Preprocessed into {len(tokenized_sentences)} sentences ({total_tokens} tokens).")

    # 3. Sentence-level train/validation split
    train_sents, val_sents = split_sentences_train_val(
        tokenized_sentences, val_ratio=args.val_ratio, seed=args.seed
    )
    print(
        f"[3/8] Dataset Split: {len(train_sents)} training sentences, "
        f"{len(val_sents)} held-out validation sentences."
    )

    # 4. Build vocabulary on training sentences
    vocab, word2id, id2word, counts = build_vocabulary(train_sents, min_count=args.min_count)
    vocab_size = len(vocab)
    print(f"[4/8] Vocabulary built: {vocab_size} unique words (min_count >= {args.min_count}).")

    # 5. Generate Skip-gram context pairs
    train_pairs = generate_skipgram_pairs(train_sents, word2id, window=args.window)
    val_pairs = generate_skipgram_pairs(val_sents, word2id, window=args.window)
    print(
        f"[5/8] Generated {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs."
    )

    if not train_pairs:
        raise ValueError("No training pairs generated. Please adjust min_count or max_sentences.")

    # 6. Initialize Model
    model = SkipGramModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
        init_range=0.1,
    )
    print(f"[6/8] Initialized SkipGramModel: E ({vocab_size}x{args.embedding_dim}), U ({args.embedding_dim}x{vocab_size}).")

    # Numerical gradient verification before training
    c_check, t_check = train_pairs[0]
    max_diff_U, max_diff_h, _ = finite_difference_gradient_check(
        model, c_check, t_check, epsilon=1e-5
    )
    print(
        f"      Initial numerical gradient discrepancy: U={max_diff_U:.2e}, h={max_diff_h:.2e} (tolerance: 1e-5)."
    )

    # 7. Training loop
    print("[7/8] Commencing manual SGD training...")
    start_time = time.time()

    initial_train_loss = model.evaluate_dataset_loss(train_pairs)
    initial_val_loss = model.evaluate_dataset_loss(val_pairs) if val_pairs else None
    init_val_str = f"{initial_val_loss:.6f}" if initial_val_loss is not None else "N/A"
    print(
        f"      Epoch   0/{args.epochs} | Train Loss: {initial_train_loss:.6f} | "
        f"Val Loss: {init_val_str}"
    )

    pair_rng = random.Random(args.seed)
    loss_history = []
    val_history = []

    for epoch in range(1, args.epochs + 1):
        # Shuffle pairs each epoch
        epoch_pairs = list(train_pairs)
        pair_rng.shuffle(epoch_pairs)

        # SGD updates
        for center_id, target_id in epoch_pairs:
            model.sgd_step(center_id, target_id, learning_rate=args.learning_rate)

        # Exact dataset evaluation at fixed weights
        epoch_train_loss = model.evaluate_dataset_loss(train_pairs)
        epoch_val_loss = model.evaluate_dataset_loss(val_pairs) if val_pairs else None

        loss_history.append(epoch_train_loss)
        if epoch_val_loss is not None:
            val_history.append(epoch_val_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            val_str = f"{epoch_val_loss:.6f}" if epoch_val_loss is not None else "N/A"
            print(
                f"      Epoch {epoch:3d}/{args.epochs} | Train Loss: {epoch_train_loss:.6f} | "
                f"Val Loss: {val_str}"
            )

    training_duration = time.time() - start_time
    final_train_loss = loss_history[-1]
    final_val_loss = val_history[-1] if val_history else None

    abs_loss_reduction = initial_train_loss - final_train_loss
    pct_loss_reduction = (abs_loss_reduction / initial_train_loss) * 100.0

    print("-" * 65)
    print(f"Training completed in {training_duration:.2f} seconds ({args.epochs / training_duration:.1f} epochs/sec).")
    print(f"Initial Train Loss: {initial_train_loss:.6f}")
    print(f"Final Train Loss:   {final_train_loss:.6f}")
    print(f"Absolute Reduction: {abs_loss_reduction:.6f}")
    print(f"Relative Reduction: {pct_loss_reduction:.2f}%")
    if final_val_loss is not None:
        print(f"Initial Val Loss:   {initial_val_loss:.6f}")
        print(f"Final Val Loss:     {final_val_loss:.6f}")
    print("-" * 65)

    # 8. Nearest Neighbor Queries
    print("[8/8] Evaluating Nearest Neighbors for Tigrinya vocabulary words:")
    # Select up to 5 representative words from the vocabulary
    target_query_words = ["ሰብ", "ዓለም", "ሰላም", "ሓዲሽ", "ግዜ", "ኣብ", "እቲ", "ምስ"]
    eval_words = [w for w in target_query_words if w in word2id][:5]
    if len(eval_words) < 5:
        # Fall back to top frequency words
        eval_words = vocab[:5]

    nearest_neighbors_report = {}
    for word in eval_words:
        nn = get_nearest_neighbors(model, word, word2id, id2word, top_k=3)
        nearest_neighbors_report[word] = nn
        nn_str = ", ".join([f"{w} ({sim:+.4f})" for w, sim in nn])
        print(f"      '{word}' -> {nn_str}")

    # Test unknown word handling
    unknown_test = get_nearest_neighbors(model, "ዘይፍሉጥ_ቃል", word2id, id2word)
    print(f"      Unknown word 'ዘይፍሉጥ_ቃል' returned: {unknown_test} (graceful fallback).")

    # Serialize Model to JSON
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)

    hyperparameters = {
        "embedding_dim": args.embedding_dim,
        "window": args.window,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "seed": args.seed,
        "min_count": args.min_count,
        "max_sentences": args.max_sentences,
        "val_ratio": args.val_ratio,
    }

    corpus_metadata = {
        "dataset_name": "Gamayun Language Data Kits — Tigrinya (5k)",
        "source": "CLEAR Global / Translators without Borders",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "raw_sentences_loaded": len(raw_sentences),
        "training_sentences": len(train_sents),
        "validation_sentences": len(val_sents),
        "total_tokens_train": sum(len(s) for s in train_sents),
        "vocabulary_size": vocab_size,
        "training_pairs_count": len(train_pairs),
        "validation_pairs_count": len(val_pairs),
    }

    metrics_payload = {
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "absolute_loss_reduction": abs_loss_reduction,
        "percentage_loss_reduction": pct_loss_reduction,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "training_duration_seconds": training_duration,
        "epochs_completed": args.epochs,
        "gradient_check_max_diff_U": max_diff_U,
        "gradient_check_max_diff_h": max_diff_h,
        "loss_history_train": loss_history,
        "loss_history_val": val_history,
        "nearest_neighbors_samples": nearest_neighbors_report,
    }

    save_model(
        filepath=args.output_model,
        model=model,
        vocab=vocab,
        word2id=word2id,
        hyperparameters=hyperparameters,
        corpus_metadata=corpus_metadata,
        training_metrics=metrics_payload,
    )
    print(f"\nModel successfully saved to: {args.output_model}")

    with open(args.metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    print(f"Training metrics saved to:  {args.metrics_output}")

    # Verify saved model
    loaded_model, loaded_vocab, loaded_w2i, loaded_i2w, _ = load_model(args.output_model)
    valid, msg = verify_saved_model(model, loaded_model, vocab, loaded_vocab, tolerance=1e-5)
    print(f"Model reload integrity check: {msg} (valid: {valid})")

    return metrics_payload


if __name__ == "__main__":
    cli_args = parse_args()
    run_experiment(cli_args)
