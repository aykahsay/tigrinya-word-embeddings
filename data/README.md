# Tigrinya Corpus Documentation

This directory manages the training and evaluation corpus for the Tigrinya Skip-gram Word Embeddings project.

---

## 1. Provenance and Source

* **Dataset Title**: [TWB Parallel Sentence Kits – Tigrinya (5k)](https://mozilladatacollective.com/datasets/cmoskmbpj00vxnu07w8lu7rrk)
* **Hosting Platform**: **Mozilla Data Collective** (`https://mozilladatacollective.com/datasets/cmoskmbpj00vxnu07w8lu7rrk`)
* **Provider**: **CLEAR Global (formerly Translators without Borders)**
* **Content**: 5,000 English–Tigrinya parallel sentence pairs
* **Format**: TSV (`kit5k.tsv`) inside a `.tar.gz` archive (`1777882055452-CLEAR_Gamayun_tig-eng_5k.tar.gz`)
* **License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
  * Full unaltered license text preserved in: [`data/LICENSE_DATA.txt`](file:///c:/icoglabs/tigrinya-word-embeddings/data/LICENSE_DATA.txt)
  * Canonical license reference: [https://creativecommons.org/licenses/by/4.0/legalcode](https://creativecommons.org/licenses/by/4.0/legalcode)
* **Attribution Requirement**: **Credit CLEAR Global** when using or deriving from this dataset.
* **Academic Citation**:
  ```bibtex
  @inproceedings{oktem-etal-2020-gamayun,
    title     = {Gamayun -- Language Technology for Humanitarian Response},
    author    = {Alp {\"O}ktem and Muhannad Albayk Jaam and Eric DeLuca and Grace Tang},
    booktitle = {2020 IEEE Global Humanitarian Technology Conference (GHTC)},
    year      = {2020},
    pages     = {1--7},
    doi       = {10.1109/GHTC49095.2020.9342939}
  }
  ```

---

## 2. Dataset Overview

The dataset consists of 5,000 sentences professionally translated from English into Tigrinya by native translators. The sentences cover diverse communicative contexts (general domain, humanitarian, and conversational).

### File Details
* **Raw Archive**: `data/1777882055452-CLEAR_Gamayun_tig-eng_5k.tar.gz` (414,461 bytes)
* **Raw Parallel File**: `data/raw/kit5k.tsv` (747,409 bytes, 5,000 parallel sentence rows)
* **Target Column**: `tig` (Tigrinya translation in Ge'ez script)

---

## 3. Preprocessing Rules

Our preprocessing pipeline (`corpus.py`) applies deterministic, linguistically motivated steps tailored for Ge'ez script Tigrinya:

1. **Extraction**: Extract only the Tigrinya side (`tig`) for monolingual distributional representation learning.
2. **Unicode Normalization**: Apply canonical decomposition followed by canonical composition (`unicodedata.normalize('NFC', text)`).
3. **Sentence Boundary Detection**: Sentences are split on standard Ethiopic terminators (`።` - Ethiopic full stop, `፧` - Ethiopic question mark) as well as Latin terminators (`?`, `!`, `.`). Sentence boundaries are strictly preserved; context windows **never** cross sentence boundaries.
4. **Tokenization**:
   - Ethiopic word separators (`፡` - wordspace, `፣` - comma, `፤` - semicolon, `፥` - colon, `፦` - preface colon) and Latin punctuation are replaced by token boundaries.
   - Word tokens are extracted matching Ethiopic syllable ranges (`[\u1200-\u135A\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]+`), with optional internal contractions (e.g., `ድሕሪ'ዚ` / `ድሕሪ’ዚ`).
   - Standalone numerals and punctuation symbols are excluded from the vocabulary to focus embeddings on semantic lexical items.
5. **Vocabulary Construction**: Deterministic alphabetical sorting of vocabulary words meeting a configurable minimum frequency threshold (`min_count`).
6. **Sentence-Level Train/Validation Split**: Sentences are partitioned into training and held-out evaluation sets before extracting pairs, preventing data leakage across splits.

---

## 4. Corpus Statistics

| Metric | Raw Dataset | Processed Pilot Corpus |
| :--- | :--- | :--- |
| Raw TSV Rows | 5,000 | 600 selected sentences |
| Non-empty Tigrinya Lines | 5,000 | 600 sentences |
| Training Sentences | — | 500 sentences |
| Held-out Validation Sentences | — | 100 sentences |
| Total Tokens (Training Split) | 39,483 (full) | 3,742 tokens |
| Vocabulary Threshold (`min_count`) | — | 3 |
| Retained Vocabulary Size ($V$) | 12,501 (raw types) | 201 unique words |
| Skip-gram Pairs ($N$, window=2) | — | 2,846 training pairs |

---

## 5. Known Corpus Limitations

1. **Corpus Size**: The dataset is a focused pilot corpus designed to demonstrate mathematical correctness, gradient dynamics, and loss convergence without requiring high-performance computing clusters or numerical acceleration libraries.
2. **Morphological Complexity**: Tigrinya has an agglutinative and templatic morphology (prefixes, infixes, suffixes attached to triconsonantal roots). Without morphological lemmatization, inflected variants (e.g., `ሰብ`, `ሰባት`, `ንሰብ`) receive distinct atomic vector IDs.
3. **Domain Coverage**: General and conversational domain sentences may under-represent specialized technical, literary, or archaic Ge'ez vocabularies.
