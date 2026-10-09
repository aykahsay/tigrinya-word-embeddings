# corpus_probe_7Q4M
"""
Skip-gram Neural Network for Tigrinya Word Embeddings from Scratch.
Implements the complete forward pass, stable log-sum-exp cross entropy loss,
analytical score and parameter gradients, and SGD parameter updates.

Implemented using ONLY Python standard library (math, random).
No NumPy, PyTorch, TensorFlow, JAX, or autograd libraries.
"""

import math
import random


class SkipGramModel:
    """
    Skip-gram word embedding model using row-vector conventions.

    Parameters:
        vocab_size (int): V, number of unique words in vocabulary.
        embedding_dim (int): d, dimensionality of word representations.
        seed (int): Random seed for reproducible weight initialization.
        init_range (float): Uniform initialization boundary [-init_range, init_range].
    """

    def __init__(self, vocab_size, embedding_dim, seed=42, init_range=0.1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.init_range = init_range

        # E: Input embedding matrix of shape V x d
        # U: Output context matrix of shape d x V
        self.E = []
        self.U = []
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initializes E and U with small random floats using the recorded seed."""
        rng = random.Random(self.seed)
        self.E = [
            [rng.uniform(-self.init_range, self.init_range) for _ in range(self.embedding_dim)]
            for _ in range(self.vocab_size)
        ]
        self.U = [
            [rng.uniform(-self.init_range, self.init_range) for _ in range(self.vocab_size)]
            for _ in range(self.embedding_dim)
        ]

    def embedding_lookup(self, center_id):
        """
        Retrieves center word representation h = E[center_id].
        Shape: 1 x d (represented as a 1D Python list of length d).
        """
        return self.E[center_id]

    def compute_scores(self, h):
        """
        Computes candidate context scores s = h * U.
        h shape: 1 x d, U shape: d x V -> s shape: 1 x V.
        s[j] = sum_{k=0}^{d-1} h[k] * U[k][j].
        Implemented with explicit loops.
        """
        V = self.vocab_size
        d = self.embedding_dim
        s = [0.0] * V
        for k in range(d):
            hk = h[k]
            Uk = self.U[k]
            for j in range(V):
                s[j] += hk * Uk[j]
        return s

    @staticmethod
    def stable_softmax(scores):
        """
        Computes numerically stable softmax probabilities via the max-shift trick:
        m = max(s)
        a[j] = exp(s[j] - m)
        p[j] = a[j] / sum(a)

        Returns:
            p (list): Softmax probabilities summing to 1.0.
            m (float): Maximum score used for stabilization.
            a (list): Shifted exponentials.
            sum_a (float): Sum of shifted exponentials.
        """
        m = max(scores)
        a = [math.exp(sj - m) for sj in scores]
        sum_a = sum(a)
        p = [val / sum_a for val in a]
        return p, m, a, sum_a

    @staticmethod
    def compute_one_pair_loss(scores, target_id, m, sum_a):
        """
        Computes stable cross-entropy loss for target index t:
        L = (m - s[t]) + ln(sum(a))
        Avoids naive -ln(p[t]) which underflows when p[t] -> 0.
        """
        return (m - scores[target_id]) + math.log(sum_a)

    def forward(self, center_id, target_id):
        """
        Executes the complete forward pass for a single (center, target) pair.

        Returns:
            scores (list): 1 x V candidate scores.
            p (list): 1 x V normalized probabilities.
            loss (float): Scalar cross-entropy loss.
            h (list): 1 x d center embedding vector.
        """
        h = self.embedding_lookup(center_id)
        scores = self.compute_scores(h)
        p, m, a, sum_a = self.stable_softmax(scores)
        loss = self.compute_one_pair_loss(scores, target_id, m, sum_a)
        return scores, p, loss, h

    def compute_gradients(self, h, p, target_id):
        """
        Computes analytical gradients for output matrix U and center embedding h:
        e[j] = p[j] - 1 if j == t else p[j]
        grad_U[k][j] = h[k] * e[j]         (shape d x V)
        grad_h[k] = sum_j U[k][j] * e[j]   (shape 1 x d, uses pre-update U)

        Returns:
            grad_U (list of list): d x V gradient matrix for U.
            grad_h (list): 1 x d gradient vector for center embedding h.
            e (list): 1 x V score error vector.
        """
        V = self.vocab_size
        d = self.embedding_dim

        # Score error vector e
        e = [p[j] - 1.0 if j == target_id else p[j] for j in range(V)]

        # Gradient w.r.t U: grad_U[k][j] = h[k] * e[j]
        grad_U = [[h[k] * e[j] for j in range(V)] for k in range(d)]

        # Gradient w.r.t h: grad_h[k] = sum_j U[k][j] * e[j] (pre-update U)
        grad_h = [0.0] * d
        for k in range(d):
            Uk = self.U[k]
            val = 0.0
            for j in range(V):
                val += Uk[j] * e[j]
            grad_h[k] = val

        return grad_U, grad_h, e

    def sgd_step(self, center_id, target_id, learning_rate):
        """
        Performs one manual SGD parameter update for pair (center_id, target_id).
        Both gradients are computed using pre-update parameters before either matrix is updated.
        Only row E[center_id] is updated; all other rows of E remain unchanged.

        Returns:
            loss (float): Scalar loss prior to parameter update.
        """
        h = self.E[center_id]
        d = self.embedding_dim
        V = self.vocab_size

        # 1. Forward pass: compute scores s = h * U
        s = [0.0] * V
        for k in range(d):
            hk = h[k]
            Uk = self.U[k]
            for j in range(V):
                s[j] += hk * Uk[j]

        # 2. Stable softmax and loss
        m = max(s)
        sum_a = 0.0
        a = [0.0] * V
        for j in range(V):
            val = math.exp(s[j] - m)
            a[j] = val
            sum_a += val

        p = [val / sum_a for val in a]
        loss = (m - s[target_id]) + math.log(sum_a)

        # 3. Score error vector e = p - y
        e = list(p)
        e[target_id] -= 1.0

        # 4. Compute gradients and apply SGD updates
        # Uk[j] is read to compute grad_h before being updated in place
        center_row = self.E[center_id]
        for k in range(d):
            hk = h[k]
            Uk = self.U[k]
            gh_k = 0.0
            for j in range(V):
                ej = e[j]
                gh_k += Uk[j] * ej
                Uk[j] -= learning_rate * (hk * ej)
            center_row[k] -= learning_rate * gh_k

        return loss

    def evaluate_dataset_loss(self, pairs):
        """
        Evaluates mean dataset loss across a collection of (center_id, target_id) pairs
        at fixed weights without performing parameter updates:
        J = (1 / N) * sum_{n=1}^N L_n
        """
        if not pairs:
            return 0.0

        d = self.embedding_dim
        V = self.vocab_size
        cache = {}
        total_loss = 0.0

        for center_id, target_id in pairs:
            if center_id not in cache:
                h = self.E[center_id]
                s = [0.0] * V
                for k in range(d):
                    hk = h[k]
                    Uk = self.U[k]
                    for j in range(V):
                        s[j] += hk * Uk[j]
                m = max(s)
                sum_a = sum(math.exp(sj - m) for sj in s)
                ln_sum_a = math.log(sum_a)
                cache[center_id] = (s, m, ln_sum_a)

            s, m, ln_sum_a = cache[center_id]
            loss = (m - s[target_id]) + ln_sum_a
            total_loss += loss

        return total_loss / len(pairs)
