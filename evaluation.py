"""
Evaluation utilities for Tigrinya Skip-gram embeddings.
Implements manual cosine similarity, nearest-neighbor queries,
held-out loss evaluation, and numerical finite-difference gradient checks.

Pure Python standard library implementation.
"""

import math


def vector_norm(vec):
    """Computes Euclidean (L2) norm of a vector using explicit loops."""
    sum_sq = 0.0
    for val in vec:
        sum_sq += val * val
    return math.sqrt(sum_sq)


def vector_dot(vec_a, vec_b):
    """Computes dot product of two vectors of equal dimensionality."""
    total = 0.0
    for a, b in zip(vec_a, vec_b):
        total += a * b
    return total


def cosine_similarity(vec_a, vec_b):
    """
    Computes cosine similarity between two vectors:
    cosine(a, b) = sum_k(a[k] * b[k]) / (sqrt(sum_k a[k]^2) * sqrt(sum_k b[k]^2))

    Handles zero-norm vectors safely without crashing by returning 0.0.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vector dimensionalities must match for cosine similarity.")

    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot = vector_dot(vec_a, vec_b)
    return dot / (norm_a * norm_b)


def get_nearest_neighbors(model, query_word, word2id, id2word, top_k=3):
    """
    Finds the top_k most similar vocabulary words for query_word based on
    cosine similarity between input embedding vectors in E.
    Excludes the query word itself.

    Returns:
        neighbors (list of tuple): List of (word, cosine_score) sorted descending,
        or None if query_word is out-of-vocabulary.
    """
    if query_word not in word2id:
        return None

    query_id = word2id[query_word]
    query_vec = model.embedding_lookup(query_id)

    similarities = []
    for other_id in range(model.vocab_size):
        if other_id == query_id:
            continue
        other_word = id2word[other_id]
        other_vec = model.embedding_lookup(other_id)
        sim = cosine_similarity(query_vec, other_vec)
        similarities.append((other_word, sim))

    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_k]


def finite_difference_gradient_check(model, center_id, target_id, epsilon=1e-5):
    """
    Performs central finite-difference gradient verification:
    grad_num = [L(theta + epsilon) - L(theta - epsilon)] / (2 * epsilon)

    Verifies analytical gradients for output matrix U and center embedding vector h.

    Returns:
        max_diff_U (float): Maximum absolute discrepancy across all entries of U.
        max_diff_h (float): Maximum absolute discrepancy across entries of center row E[center_id].
        details (dict): Diagnostic details including analytical and numerical values.
    """
    d = model.embedding_dim
    V = model.vocab_size

    # Analytical gradients
    scores, p, loss, h = model.forward(center_id, target_id)
    grad_U_ana, grad_h_ana, _ = model.compute_gradients(h, p, target_id)

    # Numerical check for U
    grad_U_num = [[0.0] * V for _ in range(d)]
    for k in range(d):
        for j in range(V):
            orig = model.U[k][j]

            model.U[k][j] = orig + epsilon
            _, _, loss_plus, _ = model.forward(center_id, target_id)

            model.U[k][j] = orig - epsilon
            _, _, loss_minus, _ = model.forward(center_id, target_id)

            model.U[k][j] = orig  # restore
            grad_num = (loss_plus - loss_minus) / (2.0 * epsilon)
            grad_U_num[k][j] = grad_num

    # Numerical check for h (row E[center_id])
    grad_h_num = [0.0] * d
    for k in range(d):
        orig = model.E[center_id][k]

        model.E[center_id][k] = orig + epsilon
        _, _, loss_plus, _ = model.forward(center_id, target_id)

        model.E[center_id][k] = orig - epsilon
        _, _, loss_minus, _ = model.forward(center_id, target_id)

        model.E[center_id][k] = orig  # restore
        grad_num = (loss_plus - loss_minus) / (2.0 * epsilon)
        grad_h_num[k] = grad_num

    max_diff_U = max(abs(grad_U_ana[k][j] - grad_U_num[k][j]) for k in range(d) for j in range(V))
    max_diff_h = max(abs(grad_h_ana[k] - grad_h_num[k]) for k in range(d))

    return max_diff_U, max_diff_h, {
        "grad_U_ana": grad_U_ana,
        "grad_U_num": grad_U_num,
        "grad_h_ana": grad_h_ana,
        "grad_h_num": grad_h_num,
    }
