"""
Tigrinya Corpus Preprocessing and Skip-Gram Training Pair Generation.
Pure Python standard library implementation (no third-party dependencies).
"""

import csv
import collections
import re
import unicodedata
import random


ETHIOPIC_WORD_REGEX = re.compile(
    r"[\u1200-\u135A\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]+(?:['\u2019][\u1200-\u135A\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]+)?"
)

SENTENCE_SPLIT_REGEX = re.compile(r"[።፧\?!.]+")


def normalize_tigrinya_text(text):
    """Applies canonical Unicode NFC normalization to Tigrinya text."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text.strip())


def segment_sentences(text):
    """
    Splits text into sentence strings using Ethiopic (።, ፧) and
    standard (?, !, .) sentence terminators.
    """
    normalized = normalize_tigrinya_text(text)
    if not normalized:
        return []
    segments = SENTENCE_SPLIT_REGEX.split(normalized)
    return [s.strip() for s in segments if s.strip()]


def tokenize_sentence(sentence):
    """
    Extracts Tigrinya Ge'ez words from a single sentence string.
    Punctuation and isolated digits are discarded.
    """
    words = ETHIOPIC_WORD_REGEX.findall(sentence)
    return [w for w in words if w]


def preprocess_corpus_lines(raw_lines):
    """
    Preprocesses a list of raw Tigrinya text lines into a list of
    tokenized sentences (list of list of str).
    Preserves strict sentence boundaries.
    """
    tokenized_sentences = []
    for line in raw_lines:
        for sent in segment_sentences(line):
            tokens = tokenize_sentence(sent)
            if len(tokens) >= 2:  # Sentences need at least 2 words for context pairs
                tokenized_sentences.append(tokens)
    return tokenized_sentences


def load_raw_corpus_tsv(tsv_path, tig_col_index=1, limit=None):
    """
    Loads raw Tigrinya sentences from a TSV file (e.g. Gamayun kit5k.tsv).
    """
    sentences = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) > tig_col_index:
                tig = row[tig_col_index].strip()
                if tig:
                    sentences.append(tig)
                    if limit and len(sentences) >= limit:
                        break
    return sentences


def build_vocabulary(tokenized_sentences, min_count=1):
    """
    Builds a deterministic vocabulary from tokenized sentences.
    Sorts vocabulary words alphabetically to ensure strict determinism.

    Returns:
        vocab (list): Alphabetically sorted list of unique words.
        word2id (dict): Mapping from word str to integer ID.
        id2word (dict): Mapping from integer ID to word str.
    """
    counts = collections.Counter()
    for sentence in tokenized_sentences:
        counts.update(sentence)

    filtered_words = [w for w, c in counts.items() if c >= min_count]
    vocab = sorted(filtered_words)

    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for i, w in enumerate(vocab)}

    return vocab, word2id, id2word, counts


def extract_context_pairs(sentence, window=2):
    """
    Generates (center_word, context_word) pairs for a single sentence
    given a symmetric context window size.
    Never crosses sentence boundaries.

    For sentence [a, b, c, d] and window=1, produces:
    (a, b), (b, a), (b, c), (c, b), (c, d), (d, c).
    """
    pairs = []
    length = len(sentence)
    for i in range(length):
        center = sentence[i]
        start = max(0, i - window)
        end = min(length, i + window + 1)
        for j in range(start, end):
            if i != j:
                pairs.append((center, sentence[j]))
    return pairs


def generate_skipgram_pairs(tokenized_sentences, word2id, window=2):
    """
    Converts tokenized sentences into integer (center_id, target_id) training pairs.
    Tokens not in word2id (below min_count) are filtered prior to context window creation.
    Repeated pair occurrences are preserved to reflect natural word frequency.
    """
    all_pairs = []
    for sentence in tokenized_sentences:
        filtered_ids = [word2id[w] for w in sentence if w in word2id]
        if len(filtered_ids) >= 2:
            sentence_pairs = extract_context_pairs(filtered_ids, window=window)
            all_pairs.extend(sentence_pairs)
    return all_pairs


def split_sentences_train_val(tokenized_sentences, val_ratio=0.15, seed=42):
    """
    Splits tokenized sentences into training and validation sets
    at the sentence level to prevent pair leakage.
    """
    shuffled = list(tokenized_sentences)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_count = int(len(shuffled) * val_ratio)
    val_sentences = shuffled[:val_count]
    train_sentences = shuffled[val_count:]

    return train_sentences, val_sentences
