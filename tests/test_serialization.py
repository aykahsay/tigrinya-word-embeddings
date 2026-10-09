"""
Unit tests for JSON model saving, reloading, and numerical integrity.
Validates Check 8 from the specification (tolerance 0.00001).
"""

import os
import tempfile
import unittest
from skipgram_tigrinya import SkipGramModel
from serialization import save_model, load_model, verify_saved_model
from evaluation import get_nearest_neighbors


class TestSerialization(unittest.TestCase):

    def test_save_and_reload_model_integrity(self):
        """
        Verify saved model: vocabulary, weights, nearest-neighbor order,
        and softmax probabilities match the original within tolerance.
        """
        vocab = ["ሰብ", "ዓለም", "ሰላም", "ሓዲሽ", "ጊዜ"]
        word2id = {w: i for i, w in enumerate(vocab)}
        id2word = {i: w for i, w in enumerate(vocab)}

        model = SkipGramModel(vocab_size=len(vocab), embedding_dim=4, seed=42)

        # Train a few steps to introduce non-symmetric weights
        for c, t in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            model.sgd_step(c, t, learning_rate=0.05)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_model(
                filepath=tmp_path,
                model=model,
                vocab=vocab,
                word2id=word2id,
                hyperparameters={"embedding_dim": 4, "seed": 42},
                corpus_metadata={"source": "test_corpus"},
            )

            loaded_model, loaded_vocab, loaded_w2i, loaded_i2w, loaded_meta = load_model(tmp_path)

            # Verify integrity using verification utility
            valid, msg = verify_saved_model(model, loaded_model, vocab, loaded_vocab, tolerance=1e-5)
            self.assertTrue(valid, f"Model verification failed: {msg}")

            # Verify nearest neighbor order matches
            orig_nn = get_nearest_neighbors(model, "ሰብ", word2id, id2word, top_k=3)
            load_nn = get_nearest_neighbors(loaded_model, "ሰብ", loaded_w2i, loaded_i2w, top_k=3)

            self.assertEqual([w for w, _ in orig_nn], [w for w, _ in load_nn])
            for (_, s_orig), (_, s_load) in zip(orig_nn, load_nn):
                self.assertAlmostEqual(s_orig, s_load, places=5)

            # Test unknown word handling
            unknown_nn = get_nearest_neighbors(loaded_model, "ዘይፍሉጥ", loaded_w2i, loaded_i2w)
            self.assertIsNone(unknown_nn)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
