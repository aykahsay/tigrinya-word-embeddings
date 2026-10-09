"""
Unit tests for analytical gradients and numerical finite-difference checks.
Validates Check 3 from the specification (tolerance 0.00001).
"""

import unittest
from skipgram_tigrinya import SkipGramModel
from evaluation import finite_difference_gradient_check


class TestGradients(unittest.TestCase):

    def setUp(self):
        self.model = SkipGramModel(vocab_size=3, embedding_dim=2, seed=42)
        self.model.E = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        self.model.U = [[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]

    def test_check3_analytical_gradients(self):
        """
        Check 3: Gradients
        Expected grad_U:
        [[0.244728, -0.334759, 0.090031],
         [0, 0, 0]]
        Expected grad_h:
        [-0.424790, 0.334759]
        Tolerance: 0.00001
        """
        center_id = 0  # cats
        target_id = 1  # eat

        scores, p, loss, h = self.model.forward(center_id, target_id)
        grad_U, grad_h, e = self.model.compute_gradients(h, p, target_id)

        expected_grad_U = [
            [0.244728, -0.334759, 0.090031],
            [0.0, 0.0, 0.0],
        ]
        expected_grad_h = [-0.424790, 0.334759]

        for k in range(2):
            for j in range(3):
                self.assertAlmostEqual(grad_U[k][j], expected_grad_U[k][j], places=5)

        for k in range(2):
            self.assertAlmostEqual(grad_h[k], expected_grad_h[k], places=5)

    def test_numerical_gradient_finite_difference(self):
        """
        Verify analytical gradients against central finite differences:
        numerical_gradient = [L(theta + eps) - L(theta - eps)] / (2 * eps)
        with eps = 0.00001.
        Tolerance: max error < 0.00001.
        """
        center_id = 0
        target_id = 1
        epsilon = 1e-5

        max_diff_U, max_diff_h, _ = finite_difference_gradient_check(
            self.model, center_id, target_id, epsilon=epsilon
        )

        self.assertLess(max_diff_U, 1e-5, f"Discrepancy in grad_U ({max_diff_U}) exceeds 1e-5")
        self.assertLess(max_diff_h, 1e-5, f"Discrepancy in grad_h ({max_diff_h}) exceeds 1e-5")


if __name__ == "__main__":
    unittest.main()
