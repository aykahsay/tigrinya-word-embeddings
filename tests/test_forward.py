"""
Unit tests for forward pass, scoring, softmax probabilities, and loss.
Validates Check 2 from the specification (tolerance 0.00001).
"""

import unittest
from skipgram_tigrinya import SkipGramModel


class TestForwardPass(unittest.TestCase):

    def setUp(self):
        # Vocabulary: [cats, eat, food] -> V = 3, d = 2
        self.vocab = ["cats", "eat", "food"]
        self.model = SkipGramModel(vocab_size=3, embedding_dim=2, seed=42)

        # Set specification weights
        # E: [[1, 0], [0, 1], [1, 1]]
        # U: [[0, 1, -1], [1, 0, 1]]
        self.model.E = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        self.model.U = [[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]

    def test_check2_scores_probabilities_loss(self):
        """
        Check 2: Forward pass
        Center: cats (i=0), Target: eat (t=1).
        Expected scores: [0, 1, -1]
        Expected probabilities: [0.244728, 0.665241, 0.090031]
        Expected loss: 0.407606
        Tolerance: 0.00001
        """
        center_id = 0  # cats
        target_id = 1  # eat
        scores, p, loss, h = self.model.forward(center_id, target_id)

        # 1. Check scores
        expected_scores = [0.0, 1.0, -1.0]
        for sj, exp_sj in zip(scores, expected_scores):
            self.assertAlmostEqual(sj, exp_sj, places=5)

        # 2. Check probabilities
        expected_p = [0.244728, 0.665241, 0.090031]
        for pj, exp_pj in zip(p, expected_p):
            self.assertAlmostEqual(pj, exp_pj, places=5)
            self.assertGreaterEqual(pj, 0.0)

        # Check probabilities sum to 1.0
        self.assertAlmostEqual(sum(p), 1.0, places=5)

        # 3. Check loss
        expected_loss = 0.407606
        self.assertAlmostEqual(loss, expected_loss, places=5)


if __name__ == "__main__":
    unittest.main()
