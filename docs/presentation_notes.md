# Technical Presentation Outline: Tigrinya Word Embeddings from Scratch

A structured guide for presenting the mathematics, implementation, empirical training, and linguistic findings of the pure-Python Tigrinya Skip-gram project.

---

## Slide 1: Title and Research Identity
* **Title**: Tigrinya Word Embeddings from Scratch: A Transparent, Dependency-Free Neural Implementation
* **Presenter**: Ayk (aykgeh)
* **Core Question**: *Can a transparent, manually implemented Skip-gram neural network learn useful distributional representations of Tigrinya words from a small, documented corpus without neural-network libraries?*
* **Two Key Contributions**:
  1. *Pedagogical & Mathematical*: Complete derivation and pure-Python standard-library implementation of the Skip-gram learning cycle.
  2. *Empirical & Linguistic*: Training, validation, and nearest-neighbor evaluation on a legally documented Tigrinya corpus.

---

## Slide 2: Motivation — Why Tigrinya Word Embeddings Matter
* **Linguistic Background**:
  * Tigrinya (ትግርኛ) is an Afroasiatic Semitic language spoken by over 9 million people in Eritrea and northern Ethiopia.
  * Written in the ancient Ge'ez (Ethiopic) fidel syllabary (e.g., ሃ, ሁ, ሂ, ሃ, ሄ, ህ, ሆ).
  * Highly morphologically complex with non-concatenative root-and-pattern (templatic) morphology, extensive prefixing, and suffixing.
* **Low-Resource Dilemma**:
  * Scarcity of curated, standardized, open-access text benchmarks.
  * Commercial NLP frameworks overlook Semitic Ge'ez-script nuances.
  * Word vectors serve as the foundational bedrock for machine translation, sentiment analysis, search, and linguistic tools.

---

## Slide 3: Prior Work & Computational State-of-the-Art
* **FastText Tigrinya (`Hailay/fasttext-tigrinya`)**:
  * 300-dimensional subword embeddings with character n-grams (2–5) addressing out-of-vocabulary inflections.
* **Geez Word2Vec Skip-gram (`Hailay/Geez_word2vec_skipgram.model`)**:
  * 100-dimensional Skip-gram model trained on TIGQA/HornMT via Gensim.
* **HornVecs (`fgaim/HornVecs`)**:
  * Specialized FastText fork for Horn of Africa languages by Fitsum Gaim.
* **Tigrinya Pretrained Language Models (`fgaim/Tigrinya-PLMs`)**:
  * TiRoBERTa (125M), TiBERT (110M), TiELECTRA (14M) presented at EMNLP WiNLP 2021; state-of-the-art on POS tagging (95.5%) and sentiment (84.8%).
* **Distinction**:
  * Prior work focuses on high-throughput C++/PyTorch libraries. Our project focuses on total mechanistic transparency from foundational linear algebra.

---

## Slide 4: The Pedagogical Problem Addressed
* Modern deep learning often treats model training as a black box:
  * `loss.backward()` hides the vector-outer-product chain rule.
  * `optimizer.step()` abstracts parameter isolation and row updates.
  * Library-level CUDA kernels hide underflow prevention and memory layouts.
* **Our Approach**:
  * Every operation is explicit: pure Python lists and loops.
  * Zero external numerical or ML dependencies (no NumPy, PyTorch, SciPy, or Gensim).
  * Complete transparency: from one-hot concept to row lookup, dot products, stable softmax, analytical gradients, and SGD.

---

## Slide 5: Skip-Gram Architecture and Mathematical Formulation
* **Input Representation**:
  * Center word index $i \to h = E[i]$ where $E \in \mathbb{R}^{V \times d}$.
* **Candidate Scoring**:
  * Context candidate scores $s = h U$ where $U \in \mathbb{R}^{d \times V}$.
  * $s[j] = \sum_{k=0}^{d-1} h[k] U[k, j]$.
* **Softmax Probability**:
  * $p[j] = \frac{\exp(s[j])}{\sum_r \exp(s[r])}$.

---

## Slide 6: Numerical Stability — The Log-Sum-Exp Trick
* **The Hazard**:
  * If scores are large (e.g. $s = [1000, 0]$), naive $\exp(1000)$ overflows to `+inf`.
  * If target score is small, naive probability rounds to $0.0$, causing $-\ln(0) = \text{domain error}$.
* **The Solution**:
  * Max subtraction: $m = \max(s)$, $a[j] = \exp(s[j] - m)$.
  * Cross-entropy: $L = (m - s[t]) + \ln\left(\sum_r a[r]\right)$.
  * Provably shift-invariant: Adding 100 to all scores preserves identical probabilities and loss.

---

## Slide 7: Derivation of Analytical Gradients
* **Score Error Vector**:
  $$\frac{\partial L}{\partial s[j]} = e[j] = p[j] - y[j] \quad \implies \quad e[t] = p[t] - 1, \quad e[j] = p[j] \ (j \neq t)$$
* **Output Matrix Gradient**:
  $$\frac{\partial L}{\partial U[k, j]} = h[k] \cdot e[j] \quad \implies \quad \nabla_U L = h^T e \in \mathbb{R}^{d \times V}$$
* **Center Embedding Gradient**:
  $$\frac{\partial L}{\partial h[k]} = \sum_{j=0}^{V-1} U[k, j] \cdot e[j] \quad \implies \quad \nabla_h L = e U^T \in \mathbb{R}^{1 \times d}$$
* **Critical Order of Computation**:
  * Both gradients must be evaluated with pre-update weights before modifying either matrix!

---

## Slide 8: The Six Required Tests and Gradient Verification
* Automated test suite validates numerical precision with tolerance $\epsilon = 10^{-5}$:
  1. **Check 1**: Window=1 on `[a, b, c, d]` $\to$ 6 exact ordered pairs; strict sentence boundaries.
  2. **Check 2**: Forward pass on toy example $\to$ scores $[0, 1, -1]$, probabilities $[0.244728, 0.665241, 0.090031]$, loss $0.407606$.
  3. **Check 3**: Gradients $\to \nabla U$ and $\nabla h$ match specification.
  4. **Check 4**: One SGD update ($\eta=0.1$) $\to E[\text{cats}] \approx [1.042479, -0.033476]$, recomputed loss $0.361859$, other rows unchanged.
  5. **Check 5**: Extreme scores $[1000, 0] \to$ stable loss $1000.0$.
  6. **Check 6**: Shift invariance $+100 \to$ loss and probabilities unchanged.
* **Central Finite Differences**:
  $$\frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon} \quad \implies \quad \text{Discrepancy} < 10^{-10}$$

---

## Slide 9: Corpus Provenance and Training Dynamics
* **Corpus**:
  * CLEAR Global / Translators without Borders Gamayun Tigrinya Dataset (`kit5k.tsv`).
  * License: Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Setup**:
  * 600 segmented sentences, 195 vocabulary words ($min\_count \ge 3$).
  * 2,984 training pairs, 350 held-out validation pairs.
  * $d=10$, window=2, learning rate $\eta = 0.05$, 100 epochs.
* **Loss Dynamics**:
  * Initial train loss evaluated at fixed weights vs final converged loss.
  * Held-out validation tracking confirms convergence without severe divergence.

---

## Slide 10: Nearest-Neighbor Exploration & Linguistic Observations
* Cosine similarity queries for frequent Tigrinya words (`ሰብ`, `ዓለም`, `ሰላም`, `ሓዲሽ`, `ግዜ`).
* Observation on small corpora:
  * Distributional co-occurrence reflects sentence topics rather than purely refined semantic synsets.
  * Higher frequency tokens exhibit more stable relational neighborhoods.
  * Demonstrates that lower loss alone is not synonymous with complete semantic mastery.

---

## Slide 11: Limitations and Future Directions
1. **Morphological Subwords**: Pure word-level Skip-gram treats every morphological inflection as distinct. Incorporating subword n-grams (as in HornVecs/FastText) would expand vocabulary coverage.
2. **Negative Sampling**: Hierarchical softmax or negative sampling would scale the pure-Python implementation to larger corpora ($V > 10,000$).
3. **Corpus Scale**: Expanding from pilot scale to millions of tokens.

---

## Slide 12: Project Resources & Publication
* **GitHub Repository**: `https://github.com/aykgeh/tigrinya-word-embeddings-from-scratch`
* **Hugging Face Model Hub**: `https://huggingface.co/aykgeh/tigrinya-skipgram-embeddings`
* **Artifacts**: Complete JSON model weights, vocabulary mapping, metrics, and documentation.
