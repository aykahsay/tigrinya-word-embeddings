"""
Unit tests for context window and Skip-gram pair extraction.
Validates Check 1 from the specification.
"""

import unittest
from corpus import extract_context_pairs, generate_skipgram_pairs


class TestContextPairs(unittest.TestCase):

    def test_check1_specification_window1(self):
        """
        Check 1: Context window
        Input: Sentence = [a, b, c, d], window = 1.
        Expected pairs:
        (a, b), (b, a), (b, c), (c, b), (c, d), (d, c).
        """
        sentence = ["a", "b", "c", "d"]
        expected = [
            ("a", "b"),
            ("b", "a"),
            ("b", "c"),
            ("c", "b"),
            ("c", "d"),
            ("d", "c"),
        ]
        pairs = extract_context_pairs(sentence, window=1)
        self.assertEqual(pairs, expected)

    def test_window_size_2(self):
        """Test symmetric window of 2."""
        sentence = ["w0", "w1", "w2", "w3"]
        # w0: w1, w2
        # w1: w0, w2, w3
        # w2: w0, w1, w3
        # w3: w1, w2
        expected = [
            ("w0", "w1"), ("w0", "w2"),
            ("w1", "w0"), ("w1", "w2"), ("w1", "w3"),
            ("w2", "w0"), ("w2", "w1"), ("w2", "w3"),
            ("w3", "w1"), ("w3", "w2"),
        ]
        pairs = extract_context_pairs(sentence, window=2)
        self.assertEqual(pairs, expected)

    def test_no_cross_sentence_pairs(self):
        """Ensure context pairs never bridge across sentence boundaries."""
        sentences = [["s1_a", "s1_b"], ["s2_c", "s2_d"]]
        word2id = {"s1_a": 0, "s1_b": 1, "s2_c": 2, "s2_d": 3}
        pairs = generate_skipgram_pairs(sentences, word2id, window=2)

        # Expected pairs only within each sentence
        expected = [(0, 1), (1, 0), (2, 3), (3, 2)]
        self.assertEqual(pairs, expected)
        # Verify no pair connects (0, 2) or (1, 2)
        for c, t in pairs:
            self.assertFalse((c in [0, 1] and t in [2, 3]) or (c in [2, 3] and t in [0, 1]))

    def test_repeated_words_retained(self):
        """Frequency is meaningful; repeated pairs must not be deduplicated."""
        sentence = ["a", "b", "a", "b"]
        pairs = extract_context_pairs(sentence, window=1)
        # (a, b) occurs multiple times
        count_ab = pairs.count(("a", "b"))
        self.assertGreater(count_ab, 1)

    def test_empty_and_short_sentences(self):
        """Empty sentences and single-word sentences produce no pairs."""
        self.assertEqual(extract_context_pairs([], window=1), [])
        self.assertEqual(extract_context_pairs(["single"], window=1), [])


if __name__ == "__main__":
    unittest.main()
