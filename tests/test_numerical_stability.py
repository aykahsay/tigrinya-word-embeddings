"""
Unit tests for numerical stability: extreme score underflow prevention and shift invariance.
Validates Check 5 and Check 6 from the specification (tolerance 0.00001).
"""

import unittest
from skipgram_tigrinya import SkipGramModel


class TestNumericalStability(unittest.TestCase):

    def test_check5_extreme_scores_underflow_prevention(self):
        """
        Check 5: Extreme scores
        Scores = [1000, 0], target index = 1.
        Expected stable loss: 1000.
        Naive -ln(p_target) underflows to -ln(0) -> math domain error.
        Our stable formulation L = (m - s[t]) + ln(sum(a)) avoids underflow.
        """
        scores = [1000.0, 0.0]
        target_id = 1

        p, m, a, sum_a = SkipGramModel.stable_softmax(scores)
        loss = SkipGramModel.compute_one_pair_loss(scores, target_id, m, sum_a)

        self.assertAlmostEqual(loss, 1000.0, places=5)
        # Ensure target probability is >= 0 and sum of p is 1
        self.assertGreaterEqual(p[1], 0.0)
        self.assertAlmostEqual(sum(p), 1.0, places=5)

    def test_check6_shift_invariance(self):
        """
        Check 6: Shift invariance
        Add 100 to every score: probabilities and loss remain equal within tolerance.
        """
        original_scores = [0.26, 1.13, 0.60, 0.85, 0.32]
        shifted_scores = [s + 100.0 for s in original_scores]
        target_id = 2

        p_orig, m_orig, a_orig, sum_a_orig = SkipGramModel.stable_softmax(original_scores)
        loss_orig = SkipGramModel.compute_one_pair_loss(original_scores, target_id, m_orig, sum_a_orig)

        p_shift, m_shift, a_shift, sum_a_shift = SkipGramModel.stable_softmax(shifted_scores)
        loss_shift = SkipGramModel.compute_one_pair_loss(shifted_scores, target_id, m_shift, sum_a_shift)

        # Probabilities must be identical within tolerance
        for po, ps in zip(p_orig, p_shift):
            self.assertAlmostEqual(po, ps, places=5)

        # Loss must be identical within tolerance
        self.assertAlmostEqual(loss_orig, loss_shift, places=5)


if __name__ == "__main__":
    unittest.main()
