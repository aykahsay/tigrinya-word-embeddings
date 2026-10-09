# Tigrinya Word Embeddings from Scratch

[![Python Standard Library](https://img.shields.io/badge/Dependencies-Standard%20Library%20Only-blue.svg)](https://docs.python.org/3/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](data/LICENSE_DATA.txt)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-aykgeh%2Ftigrinya--skipgram--embeddings-orange)](https://huggingface.co/aykgeh/tigrinya-skipgram-embeddings)

An educational, fully transparent, and mathematically rigorous implementation of a **Skip-gram Neural Network for Tigrinya (ትግርኛ) Word Embeddings**, built entirely from scratch using only Python's standard library.

> **Zero Third-Party ML Dependencies**: No NumPy, PyTorch, TensorFlow, Keras, JAX, SciPy, Scikit-learn, Gensim, Hugging Face Transformers, or automatic differentiation libraries. Every row lookup, matrix dot product, numerically stable softmax probability, analytical gradient, and Stochastic Gradient Descent (SGD) update is implemented using explicit Python loops and lists.

---

## 1. Project Overview & Motivation

Tigrinya is a low-resource Semitic language of the Afroasiatic family, spoken by over 9 million people across Eritrea and northern Ethiopia. It is written in the Ge'ez (Ethiopic) fidel syllabary and exhibits rich, non-concatenative root-and-pattern morphology.

While modern NLP frameworks rely on heavily abstracted black-box engines, this project demonstrates:
1. **Mathematical Mechanics**: How predicting surrounding context words updates continuous dense vectors via explicit linear algebra and multivariate calculus.
2. **Empirical Grounding**: Training and evaluating the model on authentic, legally documented Tigrinya text under Creative Commons Attribution 4.0 International (CC BY 4.0).
3. **Reproducibility**: Complete unit testing across six formal analytical checks with tolerance $\epsilon = 10^{-5}$ and central finite-difference gradient verification.

### Research Question
> *Can a transparent, manually implemented Skip-gram neural network learn useful distributional representations of Tigrinya words from a small, documented corpus, and how do those representations compare with an existing Tigrinya embedding model?*

---

## 2. Prior Work & Computational Context

For an in-depth survey of existing Tigrinya computational resources, see [docs/prior_work.md](docs/prior_work.md).

* **FastText Tigrinya (`Hailay/fasttext-tigrinya`)**: 300-dimensional subword embeddings with character n-grams (2–5) addressing out-of-vocabulary inflections.
* **Geez Word2Vec Skip-gram (`Hailay/Geez_word2vec_skipgram.model`)**: 100-dimensional Skip-gram model trained on TIGQA/HornMT via Gensim.
* **HornVecs (`fgaim/HornVecs`)**: FastText fork customized for Horn of Africa languages by Fitsum Gaim.
* **Tigrinya Pretrained Language Models (`fgaim/Tigrinya-PLMs`)**: Contextual Transformer LMs (TiRoBERTa, TiBERT, TiELECTRA) by Gaim et al. (EMNLP WiNLP 2021).

Unlike prior work using compiled C++ or PyTorch backends, this project provides a transparent, pedagogical reference implementation of the learning equations.

---

## 3. Mathematical Architecture

Full mathematical derivations are detailed in [docs/methodology.md](docs/methodology.md).

Row-vector notation is used throughout:
* Vocabulary size: $V$, Embedding dimension: $d$.
* Input embedding matrix: $E \in \mathbb{R}^{V \times d}$.
* Output context matrix: $U \in \mathbb{R}^{d \times V}$.

### Operations
1. **Embedding Lookup**: $h = E[i]$ (shape $1 \times d$).
2. **Candidate Scores**: $s = h U$ (shape $1 \times V$), where $s[j] = \sum_{k=0}^{d-1} h[k] U[k, j]$.
3. **Numerically Stable Softmax**:
   $$m = \max(s), \quad a[j] = \exp(s[j] - m), \quad p[j] = \frac{a[j]}{\sum_{r=0}^{V-1} a[r]}$$
4. **Stable Cross-Entropy Loss**:
   $$L = (m - s[t]) + \ln\left(\sum_{r=0}^{V-1} a[r]\right)$$
5. **Analytical Gradients**:
   * Score error: $e[j] = p[j] - 1$ if $j = t$, else $p[j]$.
   * Output matrix gradient: $\nabla_U L = h^T e \in \mathbb{R}^{d \times V}$.
   * Center embedding gradient: $\nabla_h L = e U^T \in \mathbb{R}^{1 \times d}$ (computed with pre-update $U$).
6. **SGD Updates**:
   $$U \leftarrow U - \eta \nabla_U L, \quad E[i] \leftarrow E[i] - \eta \nabla_h L$$
   (All other rows of $E$ remain strictly unmodified during this step).

---

## 4. Dataset & Preprocessing

* **Source**: CLEAR Global / Translators without Borders Gamayun Language Data Kits — Tigrinya (`kit5k.tsv`).
* **License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)** (see [data/LICENSE_DATA.txt](data/LICENSE_DATA.txt) and [data/README.md](data/README.md)).
* **Preprocessing Rules**:
  * Sentence splitting strictly adheres to Ethiopic (`።`, `፧`) and Latin (`?`, `!`, `.`) terminators.
  * Context pairs **never cross sentence boundaries**.
  * Ge'ez words (`\u1200-\u135A\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F`) are extracted with canonical NFC normalization.
  * Deterministic vocabulary ordering with configurable frequency threshold (`min_count >= 3`).
  * Sentence-level split: 85% training, 15% held-out validation.

---

## 5. Verification Test Suite

Every component is validated against reference benchmarks before training:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

All 12 unit tests verify:
* **Check 1 (Context Pairs)**: Ordered pairs for sentence `[a,b,c,d]` with window 1.
* **Check 2 (Forward Pass)**: Reference scores $[0, 1, -1]$, probabilities, and loss $0.407606$.
* **Check 3 (Analytical Gradients)**: Reference $\nabla U$ and $\nabla h$.
* **Central Finite Differences**: $|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}| < 10^{-10}$ with $\epsilon = 10^{-5}$.
* **Check 4 (One SGD Update)**: $E[\text{cats}] \approx [1.042479, -0.033476]$, recomputed loss $0.361859$.
* **Check 5 (Extreme Scores)**: Stable loss $1000.0$ for scores $[1000, 0]$ (no underflow/overflow).
* **Check 6 (Shift Invariance)**: $+100$ score shift produces identical probabilities and loss.
* **Check 8 (Serialization)**: JSON save/load preserves weights and predictions within $10^{-5}$.

---

## 6. Training Results & Empirical Evaluation

### 6.1 Convergence Metrics
The model was trained for 100 epochs on 2,984 training pairs ($d=10$, window=2, $\eta=0.05$):
* **Initial Training Loss**: 5.273133
* **Final Training Loss**: 3.623245
* **Absolute Loss Reduction**: 1.649888
* **Relative Loss Reduction**: 31.29%
* **Training Duration**: 140.18 seconds (~0.7 epochs/sec in pure Python)
* **Initial Validation Loss**: 5.272466
* **Final Validation Loss**: 5.946779

### 6.2 Nearest-Neighbor Comparison with Pretrained Baseline
We evaluated top-3 cosine nearest neighbors on shared high-frequency Tigrinya terms against `Hailay/Geez_word2vec_skipgram.model` (100-dim Gensim Skip-gram trained on TIGQA):

| Word | Meaning | From-Scratch (Our 10-dim Model) | Pretrained Baseline (`Hailay/Geez_word2vec`) |
| :--- | :--- | :--- | :--- |
| **`ኣብ`** | in / at | `ኣብቲ` (+0.85), `ሰብ` (+0.77), `ናብ` (+0.71) | `ኣብቲ` (+0.77), `ንፈለማ` (+0.77), `ለንደን` (+0.75) |
| **`እቲ`** | the (masc) | `ብዙሕ` (+0.91), `እንግሊዘኛ` (+0.88), `እዚ` (+0.84) | `ንሱ` (+0.82), `ግድብ` (+0.81), `ኣብ'ዚ` (+0.81) |
| **`ሰብ`** | person | `ኣብ` (+0.77), `ናይ` (+0.77), `ጁልያ` (+0.75) | `ወድ` (+0.82), `ዝገብሮ` (+0.81), `ካልእ` (+0.80) |
| **`ግዜ`** | time | `መርመራ` (+0.73), `ጁልያ` (+0.73), `ዓሳ` (+0.72) | `ንመጀመርታ` (+0.94), `ፈላማይ` (+0.93), `ሕጂ` (+0.92) |
| **`ምስ`** | with | `ዳርጋ` (+0.77), `ጽባሕ` (+0.74), `ክምለስ` (+0.73) | `ኣስኳል` (+0.75), `ተስፋ` (+0.75), `ደቃ` (+0.75) |

**Key Finding**: Both models independently learn that `ኣብ` (*in/at*) and its definite inflected form `ኣብቲ` (*in the*) possess the highest semantic/grammatical similarity (+0.85 vs +0.77), demonstrating that the from-scratch network successfully extracts real distributional regularities from the Ge'ez text.

---

## 7. Installation & Reproduction

### Prerequisites
* Python 3.8+ (tested on Python 3.10)
* No third-party packages required.

### Quick Start
```bash
# 1. Clone repository
git clone https://github.com/aykgeh/tigrinya-word-embeddings-from-scratch.git
cd tigrinya-word-embeddings-from-scratch

# 2. Run test suite
python -m unittest discover -s tests -p "test_*.py" -v

# 3. Train Skip-gram model
python train.py --epochs 100 --embedding_dim 10 --window 2 --learning_rate 0.05 --seed 42
```

---

## 7. Artifacts and Citation

* Trained model JSON: `artifacts/tigrinya_skipgram_model.json`
* Detailed metrics: `reports/training_metrics.json`
* Hugging Face Model: [https://huggingface.co/aykgeh/tigrinya-skipgram-embeddings](https://huggingface.co/aykgeh/tigrinya-skipgram-embeddings)

```bibtex
@misc{tigrinya-word-embeddings-scratch-2026,
  author       = {Ayk (aykgeh)},
  title        = {Tigrinya Word Embeddings from Scratch: A Pure-Python Skip-Gram Implementation},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/aykgeh/tigrinya-word-embeddings-from-scratch}}
}
```

## License
* Source Code: [MIT License](LICENSE)
* Dataset: [Creative Commons Attribution 4.0 International (CC BY 4.0)](data/LICENSE_DATA.txt)
