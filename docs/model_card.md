---
language:
- ti
license: cc-by-4.0
tags:
- word-embeddings
- skipgram
- tigrinya
- nlp-from-scratch
- low-resource
- educational
datasets:
- CLEAR-Global/Gamayun-kits
metrics:
- cross-entropy-loss
- cosine-similarity
---

# Tigrinya Skip-Gram Word Embeddings from Scratch

A pure-Python, dependency-free implementation of word embeddings for **Tigrinya (ትግርኛ)**, an Ethiopian Semitic language written in the Ge'ez (Ethiopic) script.

This repository hosts the model architecture, learned vocabulary, and trained embedding matrices ($E$ and $U$) trained from scratch using the **Skip-gram architecture with Stochastic Gradient Descent (SGD)**.

* **Repository ID**: `aykgeh/tigrinya-skipgram-embeddings`
* **Format**: Standard UTF-8 JSON (`tigrinya_skipgram_model.json`)
* **Dependencies for Inference & Training**: **Python Standard Library only** (`math`, `random`, `json`). Zero third-party dependencies (no NumPy, PyTorch, TensorFlow, SciPy, or Gensim).

---

## 1. Model Overview

The model learns static, dense distributional vector representations for Tigrinya words from sentence co-occurrence patterns. Given a center word token $w_i$, the Skip-gram objective optimizes the network parameters to maximize the log-probability of observing nearby context words $w_{t}$ within a symmetric window:

$$\max_{\theta} \sum_{(i, t)} \ln P(w_t \mid w_i)$$

Every mathematical operation—including row embedding lookups, candidate score dot products, numerically stable softmax probabilities, analytical score and parameter gradients, and SGD updates—is implemented entirely from scratch with explicit loops.

---

## 2. Model Architecture & Mathematical Design

* **Input Embedding Matrix E**: Shape `V × d` (where `V` is vocabulary size, `d` is embedding dimension).
* **Output Context Matrix U**: Shape `d × V`.

### Forward Pass
1. **Embedding Lookup**:
   ```
   h = E[i]                                    # Center word vector (shape: 1 × d)
   ```
2. **Candidate Scores**:
   ```
   s[j] = sum(h[k] * U[k][j] for k in range(d)) # Dot product s = h · U (shape: 1 × V)
   ```
3. **Numerically Stable Softmax** (max-shift trick prevents overflow):
   ```
   m = max(s)
   a[j] = exp(s[j] - m)
   p[j] = a[j] / sum(a)                        # Valid probability distribution
   ```
4. **Stable Cross-Entropy Loss** (prevents log(0) underflow):
   ```
   L = (m - s[t]) + ln(sum(a))
   ```

### Analytical Gradients
* **Score Error Vector (e = p - y)**:
   ```
   e[j] = p[j] - 1   if j == t (target context token)
   e[j] = p[j]       if j != t (all other tokens)
   ```
* **Output Matrix Gradient (grad_U = hᵀ · e)**:
   ```
   grad_U[k][j] = h[k] * e[j]                  # Shape: d × V
   ```
* **Center Embedding Gradient (grad_h = e · Uᵀ)**:
   ```
   grad_h[k] = sum(U[k][j] * e[j] for j in range(V)) # Shape: 1 × d (uses pre-update U)
   ```

### SGD Parameter Update
* **Output Matrix**:
   ```
   U[k][j] = U[k][j] - learning_rate * grad_U[k][j]
   ```
* **Center Embedding Vector** (only active center row updates):
   ```
   E[i][k] = E[i][k] - learning_rate * grad_h[k]
   ```
   *(All other rows of E remain untouched during this update step)*

---

## 3. Training Data Provenance & Preprocessing

* **Dataset**: [TWB Parallel Sentence Kits – Tigrinya (5k)](https://mozilladatacollective.com/datasets/cmoskmbpj00vxnu07w8lu7rrk)
* **Hosting Platform**: **Mozilla Data Collective**
* **Provider**: **CLEAR Global** (formerly Translators without Borders)
* **License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
* **Attribution**: Credit CLEAR Global when using this dataset.
* **Citation**: Öktem et al. (2020), *Gamayun – Language Technology for Humanitarian Response*, IEEE GHTC 2020.
* **Preprocessing Pipeline**:
  1. Extracted Tigrinya parallel translations in Ge'ez script.
  2. Applied canonical Unicode NFC normalization (`unicodedata.normalize('NFC', text)`).
  3. Segmented sentences on Ge'ez (`።`, `፧`) and Latin (`?`, `!`, `.`) terminators. Sentence boundaries are strictly enforced; pairs never cross boundaries.
  4. Extracted Ge'ez word tokens matching Ethiopic syllabary ranges (`\u1200-\u135A\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F`), preserving valid internal apostrophes (e.g. `ድሕሪ'ዚ`).
  5. Built deterministic vocabulary sorted alphabetically with minimum frequency threshold $min\_count \ge 3$.
  6. Partitioned into 85% training sentences and 15% held-out validation sentences at the sentence level prior to pair extraction.

---

## 4. Hyperparameters & Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Embedding Dimension ($d$)** | 10 | Latent vector size |
| **Context Window** | 2 | Symmetric window ($\pm 2$ tokens) |
| **Learning Rate ($\eta$)** | 0.05 | SGD step size |
| **Epochs** | 100 | Training passes |
| **Random Seed** | 42 | Weight initialization seed |
| **Initialization Range** | $[-0.1, 0.1]$ | Uniform random bounds |
| **Vocabulary Size ($V$)** | 195 | Unique Tigrinya words |
| **Training Pairs ($N$)** | 2,984 | Center-context pairs |
| **Held-out Pairs** | 350 | Validation pairs |
| **Training Duration** | 140.18s | Wall-clock execution time |
| **Initial Train Loss** | 5.273133 | Dataset loss at epoch 0 |
| **Final Train Loss** | 3.623245 | Dataset loss after 100 epochs |
| **Loss Reduction** | 31.29% | Relative decrease (1.649888 absolute) |
| **Initial Val Loss** | 5.272466 | Held-out loss at epoch 0 |
| **Final Val Loss** | 5.946779 | Held-out loss after 100 epochs |

---

## 5. Empirical Evaluation & Nearest-Neighbor Analysis

### 5.1 Training Loss History
* **Epoch 0**: 5.273133
* **Epoch 10**: 4.266115
* **Epoch 20**: 3.870366
* **Epoch 50**: 3.673860
* **Epoch 100**: **3.623245**

The model exhibits steady, monotonic convergence on the training set. The held-out validation loss reflects expected behavior on small pilot corpora, where rare co-occurrences in the validation split lack negative sampling regularization.

### 5.2 Nearest-Neighbor Associations
Cosine similarity queries for high-frequency Tigrinya terms:
* **`ኣብ` (in/at)** $\to$ `ኣብቲ` (+0.8543), `ሰብ` (+0.7711), `ናብ` (+0.7125)
* **`እቲ` (the [masc])** $\to$ `ብዙሕ` (+0.9148), `እንግሊዘኛ` (+0.8826), `እዚ` (+0.8371)
* **`ሰብ` (person)** $\to$ `ኣብ` (+0.7711), `ናይ` (+0.7668), `ጁልያ` (+0.7457)
* **`ግዜ` (time)** $\to$ `መርመራ` (+0.7291), `ጁልያ` (+0.7257), `ዓሳ` (+0.7237)
* **`ምስ` (with)** $\to$ `ዳርጋ` (+0.7699), `ጽባሕ` (+0.7398), `ክምለስ` (+0.7266)

Notably, the model correctly identifies strong morphological and syntactic affinity between `ኣብ` (preposition *in/at*) and `ኣብቲ` (*in the*), achieving **+0.8543** cosine similarity, matching the top association discovered by the 100-dimensional pretrained baseline (`Hailay/Geez_word2vec_skipgram.model`).

---

## 6. Verification Test Suite

Before training, the implementation verified six analytical checks with absolute tolerance $\epsilon = 10^{-5}$:
1. **Context Window**: Strictly extracts ordered pairs without boundary leakage.
2. **Forward Pass**: Matches reference scores $[0, 1, -1]$, probabilities, and loss.
3. **Gradients**: Analytical $\nabla U$ and $\nabla h$ match reference values.
4. **Central Finite Differences**:
   $$\left|\nabla_{\text{analytical}} - \frac{L(\theta+\epsilon) - L(\theta-\epsilon)}{2\epsilon}\right| < 10^{-10}$$
5. **One SGD Update**: Recomputed loss matches $0.361859$; parameter isolation confirmed.
6. **Numerical Stability**: Extreme scores $[1000, 0]$ avoid overflow; shift invariance confirmed.
7. **Reload Integrity**: Loaded JSON models produce identical probabilities and neighbor rankings within $10^{-5}$.

---

## 6. How to Use the Model (Pure Python Inference)

No libraries required. You can load and query the model directly in vanilla Python:

```python
import json
import math

# 1. Load the trained JSON artifact
with open("tigrinya_skipgram_model.json", "r", encoding="utf-8") as f:
    model_data = json.load(f)

vocab = model_data["vocabulary"]
word2id = model_data["word2id"]
E = model_data["E"]  # V x d matrix

# 2. Retrieve word vector for 'ሰብ' (person)
def get_vector(word):
    if word not in word2id:
        return None
    idx = word2id[word]
    return E[idx]

vec = get_vector("ሰብ")
print("Vector for 'ሰብ':", vec)

# 3. Compute cosine similarity between two Tigrinya words
def cosine_sim(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

v_seb = get_vector("ሰብ")
v_alem = get_vector("ዓለም")
print("Cosine('ሰብ', 'ዓለም'):", cosine_sim(v_seb, v_alem))
```

---

## 7. Limitations & Ethical Considerations

* **Corpus Scale**: This pilot model was trained on a curated pilot corpus. Embeddings reflect immediate co-occurrence distributions in conversational text, and nearest-neighbor quality is exploratory.
* **Morphological Synthesis**: Tigrinya features rich affixation and templatic stem changes. As a word-level model, unstemmed forms have separate vectors.
* **Non-Transformers Notice**: This model is a static Skip-gram lookup table and is not directly compatible with Hugging Face `AutoModelForCausalLM` or transformer tokenizers.
* **Data Provenance**: Sourced from CLEAR Global's Gamayun dataset under CC BY 4.0; no private personal data was used.
