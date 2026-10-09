# Prior Work on Tigrinya Word Embeddings and Language Models

This document surveys existing research, pretrained representations, and computational linguistic resources for Tigrinya (ትግርኛ), a Semitic language of the Afroasiatic family written in the Ge'ez (Ethiopic) script. This review provides academic context and establishes fair baselines for our from-scratch Skip-gram implementation.

---

## 1. Summary of Prior Work & Verified Resources

| Resource Name | Maintainer / Authors | Date | Architecture / Type | Script / Coverage | Corpus & License | Availability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[Hailay/fasttext-tigrinya](https://huggingface.co/Hailay/fasttext-tigrinya)** | Hailay Kidu Teklehaymanot | 2024 | FastText Skip-gram (subword n-grams 2–5, dim=300) | Tigrinya (`ti`), Ge'ez script | Corpus undocumented; Apache-2.0 license | Pretrained `.bin` binary available on Hugging Face |
| **[Hailay/Geez_word2vec_skipgram.model](https://huggingface.co/Hailay/Geez_word2vec_skipgram.model)** | Hailay Kidu Teklehaymanot | 2024 | Word2Vec Skip-gram (Gensim, dim=100, window=5) | Tigrinya (`ti`), Ge'ez script | Trained on TIGQA / HornMT (Zenodo 11423987); MIT license | Pretrained model, `embeddings.txt`, `vocabulary.txt` on Hugging Face |
| **[fgaim/HornVecs](https://github.com/fgaim/HornVecs)** | Fitsum Gaim | 2019–2021 | FastText fork adapted for Horn of Africa languages | Tigrinya, Amharic, Oromo, Somali | Multi-source web crawl; BSD 3-Clause license | C++/Python source code on GitHub |
| **[fgaim/Tigrinya-PLMs](https://github.com/fgaim/Tigrinya-PLMs)** | Fitsum Gaim, Wonsuk Yang, Jong C. Park | 2021 (EMNLP WiNLP) | Contextual Transformer LMs (TiRoBERTa, TiBERT, TiELECTRA) | Tigrinya (`ti`), Ge'ez script | TLMD dataset (~40M tokens, Zenodo 5139094); CC BY-NC-SA 4.0 | Models on Hugging Face, code on GitHub |

---

## 2. Detailed Resource Profiles

### 2.1 FastText Tigrinya (`Hailay/fasttext-tigrinya`)
* **URL**: [https://huggingface.co/Hailay/fasttext-tigrinya](https://huggingface.co/Hailay/fasttext-tigrinya)
* **Author**: Hailay Kidu Teklehaymanot.
* **Architecture**: FastText subword-aware Skip-gram architecture with negative sampling.
  - Dimension: 300.
  - Vocabulary size: 156,687 words.
  - Character n-grams: 2 to 5 characters.
  - Context window: 5.
  - Loss: Negative sampling (5 negative samples).
* **Representations**: Static subword embeddings. Because it computes vector sums over character n-grams, it can construct representations for out-of-vocabulary (OOV) words.
* **Corpus & Licensing**: The exact corpus and preprocessing steps are undocumented in the repository; the released binary is licensed under Apache 2.0.
* **Role in Project**: Illustrates how subword decomposition addresses the rich inflectional and derivational morphology of Ge'ez script languages.

### 2.2 Geez Word2Vec Skip-gram (`Hailay/Geez_word2vec_skipgram.model`)
* **URL**: [https://huggingface.co/Hailay/Geez_word2vec_skipgram.model](https://huggingface.co/Hailay/Geez_word2vec_skipgram.model)
* **Author**: Hailay Kidu Teklehaymanot.
* **Architecture**: Gensim Word2Vec Skip-gram (`sg=1`).
  - Dimension: 100.
  - Window size: 5.
  - Epochs: 10.
* **Corpus & Licensing**: Trained on the TIGQA question-answering corpus and HornMT translation dataset (Zenodo DOI: 10.5281/zenodo.11423986 / 11423987). Released under the MIT License.
* **Artifacts**: The repository provides both the Gensim `.model` file and exported plain text formats (`embeddings.txt` and `vocabulary.txt`).
* **Role in Project**: Serves as a direct external static Skip-gram baseline. We can inspect nearest-neighbor associations for shared high-frequency Tigrinya words without coupling to its proprietary training pipeline.

### 2.3 HornVecs (`fgaim/HornVecs`)
* **URL**: [https://github.com/fgaim/HornVecs](https://github.com/fgaim/HornVecs)
* **Author**: Fitsum Gaim.
* **Architecture**: Modified FastText engine tailored for languages of the Horn of Africa, particularly templatic root-and-pattern morphological structures.
* **License**: BSD 3-Clause (inherited from Facebook Research FastText).
* **Role in Project**: Demonstrates the motivation for non-standard tokenization and subword models in Semitic languages where root consonants interleave with vowel templates.

### 2.4 Monolingual Pretrained Language Models for Tigrinya (`fgaim/Tigrinya-PLMs`)
* **URL**: [https://github.com/fgaim/Tigrinya-PLMs](https://github.com/fgaim/Tigrinya-PLMs)
* **Authors**: Fitsum Gaim, Wonsuk Yang, Jong C. Park (KAIST).
* **Publication**: 5th WiNLP Workshop at EMNLP 2021.
* **Architecture**: Contextual transformer language models (TiRoBERTa 125M, TiBERT 110M, TiELECTRA 14M).
* **Corpus**: Tigrinya Language Modeling Dataset (TLMD), comprising ~0.5 GB of text (>40 million tokens) collected from news, blogs, and books (Zenodo: 5139094).
* **Performance**: Achieved state-of-the-art results on Tigrinya Part-of-Speech Tagging (95.49% accuracy) and Sentiment Analysis (84.76% F1).
* **Distinction**: These are deep contextual representations where a word's vector changes dynamically based on sentence context, distinct from static lookup tables.

---

## 3. Conceptual & Architectural Distinctions

### 3.1 Skip-gram vs. Continuous Bag-of-Words (CBOW)
* **Continuous Bag-of-Words (CBOW)** predicts the center word given the average or sum of the surrounding context vectors ($P(w_t \mid w_{t-c}, \dots, w_{t+c})$). It tends to smooth over rare words and trains faster on large corpora.
* **Skip-gram** flips the objective: given a single center word $w_t$, it predicts each surrounding context word $w_{t+j}$ independently ($P(w_{t+j} \mid w_t)$). Skip-gram treats each center–context occurrence as a distinct training update, allowing rare words to receive dedicated gradient signals. Our project implements the Skip-gram architecture.

### 3.2 Word2Vec vs. FastText
* **Standard Word2Vec** assigns a single, atomic vector to each vocabulary token. If a token was not encountered during training, it is entirely out-of-vocabulary (OOV) and cannot be represented.
* **FastText** decomposes each word into a bag of character $n$-grams (e.g., `<ዓር`, `ዓርክ`, `ርክኻ`, `ክኻ>`) and represents the word as the sum of its $n$-gram vectors. This provides powerful generalizations for morphological inflections, but obscures the underlying linear-algebraic mechanics of row-vector lookups and output matrix dot products.

### 3.3 Static Word Embeddings vs. Contextual Language Models
* **Static Word Embeddings** (Word2Vec, FastText, GloVe) learn fixed matrices $E \in \mathbb{R}^{V \times d}$. Each unique word has exactly one representation regardless of whether it functions as a noun, verb, or has multiple polysemous meanings.
* **Contextual Language Models** (BERT, RoBERTa, GPT) pass subword tokens through stacked Transformer self-attention blocks. A word's representation is a function of its entire linguistic context. While contextual models achieve higher task performance, static embeddings provide transparent, interpretable distributional semantics and lightweight footprints.

### 3.4 Pretrained Library Models vs. From-Scratch Implementation
Existing Tigrinya models rely on high-level optimized frameworks:
* Gensim / C++ FastText binaries compile multi-threaded C code with hierarchical softmax or negative sampling.
* Hugging Face Transformers rely on PyTorch/CUDA tensor operations and automatic differentiation.

In contrast, **our project implements the full mathematical learning cycle from scratch**:
1. Manual vector and matrix multiplications using explicit Python loops.
2. Numerically stable softmax using the log-sum-exp stabilization technique.
3. Analytical derivation of cross-entropy gradients ($\nabla U = h^T e$ and $\nabla h = e U^T$).
4. Manual parameter updates via Stochastic Gradient Descent (SGD).
5. Zero third-party numerical or neural-network dependencies (no NumPy, no PyTorch, no autograd).

---

## 4. Fair Baseline Comparison Strategy

To evaluate our model against prior work without violating reproducibility or dependency constraints:
1. We inspect shared vocabulary overlap between our corpus and the released `Hailay/Geez_word2vec_skipgram.model` vocabulary.
2. We query nearest neighbors for shared high-frequency Tigrinya terms (e.g., `ሰብ`, `ዓለም`, `ሰላም`, `ሓዲሽ`).
3. We acknowledge that our educational model operates with a small, clean corpus ($d=10$ or $d=16$) and full softmax, whereas external baselines use hundreds of thousands of sentences with negative sampling or subwords.
