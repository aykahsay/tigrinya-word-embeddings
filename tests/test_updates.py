"""
Unit tests for one SGD update step and parameter isolation.
Validates Check 4 from the specification (tolerance 0.00001).
"""

import unittest
from skipgram_tigrinya import SkipGramModel


class TestUpdates(unittest.TestCase):

    def setUp(self):
        self.model = SkipGramModel(vocab_size=3, embedding_dim=2, seed=42)
        self.model.E = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        self.model.U = [[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]

    def test_check4_one_sgd_update(self):
        """
        Check 4: One update
        Learning rate: 0.1.
        Expected E[cats]: [1.042479, -0.033476].
        Other E rows must remain unchanged.
        Recomputed loss using updated matrices: 0.361859.
        Tolerance: 0.00001.
        """
        center_id = 0  # cats
        target_id = 1  # eat
        lr = 0.1

        # Keep copy of other E rows prior to update
        row1_orig = list(self.model.E[1])
        row2_orig = list(self.model.E[2])

        # Perform one SGD step
        initial_loss = self.model.sgd_step(center_id, target_id, learning_rate=lr)
        self.assertAlmostEqual(initial_loss, 0.407606, places=5)

        # 1. Verify updated center embedding E[cats]
        expected_center = [1.042479, -0.033476]
        for val, exp_val in zip(self.model.E[0], expected_center):
            self.assertAlmostEqual(val, exp_val, places=5)

        # 2. Verify all other rows of E are untouched
        self.assertEqual(self.model.E[1], row1_orig)
        self.assertEqual(self.model.E[2], row2_orig)

        # 3. Verify recomputed loss with updated matrices
        _, _, recomputed_loss, _ = self.model.forward(center_id, target_id)
        expected_recomputed_loss = 0.361859
        self.assertAlmostEqual(recomputed_loss, expected_recomputed_loss, places=5)


if __name__ == "__main__":
    unittest.main()
