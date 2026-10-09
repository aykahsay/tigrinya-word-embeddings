# Mathematical Methodology and Derivations

This document provides a formal, step-by-step mathematical derivation of the Skip-gram architecture with negative log-likelihood (cross-entropy) loss, score error vector, analytical gradients with respect to trainable parameter matrices, numerical stabilization techniques, and Stochastic Gradient Descent (SGD) updates.

---

## 1. Architectural Notation and Model Structure

Let:
* $\mathcal{V}$ be the vocabulary of size $V = |\mathcal{V}|$.
* $d$ be the embedding dimensionality ($d \ll V$).
* $i \in \{0, 1, \dots, V-1\}$ be the index of the observed center word.
* $t \in \{0, 1, \dots, V-1\}$ be the index of the true context target word.
* $E \in \mathbb{R}^{V \times d}$ be the **input embedding matrix**, where row $E[i, :]$ represents the vector of center word $i$.
* $U \in \mathbb{R}^{d \times V}$ be the **output context matrix**, where column $U[:, j]$ represents the context vector for candidate word $j$.

We adopt row-vector notation throughout this implementation.

---

## 2. The Forward Computation Chain

### 2.1 Embedding Lookup
The center word is conceptually represented by a standard basis one-hot indicator vector $x_i \in \mathbb{R}^{1 \times V}$, where $x_i[i] = 1$ and $x_i[j] = 0$ for $j \neq i$.
The projected representation $h \in \mathbb{R}^{1 \times d}$ is given by matrix multiplication:
$$h = x_i E = E[i, :]$$
In our implementation, rather than allocating a sparse $1 \times V$ vector, we perform an explicit $O(1)$ row lookup:
$$h = E[i]$$

### 2.2 Score Computation
The raw dot-product scores $s \in \mathbb{R}^{1 \times V}$ measuring compatibility between center representation $h$ and each candidate word $j \in \mathcal{V}$ are computed as:
$$s = h U$$
For each candidate word $j \in \{0, \dots, V-1\}$:
$$s[j] = \sum_{k=0}^{d-1} h[k] \cdot U[k, j]$$

### 2.3 Numerically Stable Softmax
The conditional probability distribution over the entire vocabulary given center word $i$ is defined by the softmax function:
$$p[j] = P(w_j \mid w_i) = \frac{\exp(s[j])}{\sum_{r=0}^{V-1} \exp(s[r])}$$

#### Underflow and Overflow Hazards
In standard floating-point arithmetic:
1. If any $s[j] > 709.78$ (for IEEE 754 64-bit float), $\exp(s[j])$ overflows to `+inf`, causing the denominator to become `inf` and probabilities to become `nan`.
2. If $s[j] \ll 0$, $\exp(s[j])$ rounds to `0.0`. If the target token $t$ has a tiny probability, $p[t] = 0.0$, causing $-\ln(p[t]) \to -\ln(0) = \infty$, raising a domain error.

#### Shift Invariance Property
Because adding a constant scalar $c$ to all scores cancels out:
$$\frac{\exp(s[j] - c)}{\sum_r \exp(s[r] - c)} = \frac{\exp(s[j]) \cdot e^{-c}}{\sum_r \exp(s[r]) \cdot e^{-c}} = \frac{\exp(s[j])}{\sum_r \exp(s[r])} = p[j]$$

We set $m = \max_{j} s[j]$:
$$a[j] = \exp(s[j] - m)$$
$$p[j] = \frac{a[j]}{\sum_{r=0}^{V-1} a[r]}$$
Since $s[j] - m \le 0$ for all $j$, the exponent is bounded by 0, ensuring $\exp(s[j] - m) \in (0, 1]$ and preventing floating-point overflow.

### 2.4 Stable Cross-Entropy Loss
For a categorical target $y \in \mathbb{R}^{1 \times V}$ where $y[t] = 1$ and $y[j] = 0$ for $j \neq t$, the cross-entropy loss is:
$$L = -\sum_{j=0}^{V-1} y[j] \ln(p[j]) = -\ln(p[t])$$

Substituting the shifted softmax into $-\ln(p[t])$:
$$L = -\ln\left(\frac{a[t]}{\sum_r a[r]}\right) = -\ln(a[t]) + \ln\left(\sum_r a[r]\right)$$
Since $a[t] = \exp(s[t] - m)$, we have $\ln(a[t]) = s[t] - m$. Therefore:
$$L = (m - s[t]) + \ln\left(\sum_{r=0}^{V-1} a[r]\right)$$
This formula evaluates without computing $\ln(0)$ even when $p[t]$ would otherwise underflow.

---

## 3. Derivation of Analytical Gradients

### 3.1 Gradient with Respect to Scores ($e = \nabla_s L$)
Let $L = -\ln(p[t])$. Applying the chain rule with respect to score $s[j]$:
$$L = -s[t] + \ln\left(\sum_{r=0}^{V-1} \exp(s[r])\right)$$

Differentiating with respect to $s[j]$:
$$\frac{\partial L}{\partial s[j]} = -\frac{\partial s[t]}{\partial s[j]} + \frac{1}{\sum_r \exp(s[r])} \cdot \frac{\partial}{\partial s[j]}\left(\sum_{r=0}^{V-1} \exp(s[r])\right)$$

Notice that:
$$\frac{\partial s[t]}{\partial s[j]} = \begin{cases} 1 & \text{if } j = t \\ 0 & \text{if } j \neq t \end{cases} = y[j]$$
and:
$$\frac{\partial}{\partial s[j]} \sum_r \exp(s[r]) = \exp(s[j])$$

Therefore:
$$\frac{\partial L}{\partial s[j]} = -y[j] + \frac{\exp(s[j])}{\sum_r \exp(s[r])} = p[j] - y[j]$$

Defining the score error vector $e \in \mathbb{R}^{1 \times V}$:
$$e[j] = \begin{cases} p[j] - 1 & \text{if } j = t \\ p[j] & \text{if } j \neq t \end{cases}$$

Notice that the sum of the errors is zero:
$$\sum_{j=0}^{V-1} e[j] = \sum_{j=0}^{V-1} p[j] - 1 = 1 - 1 = 0$$

### 3.2 Gradient with Respect to Output Context Matrix $U$
Each score is $s[j] = \sum_{k=0}^{d-1} h[k] U[k, j]$. Applying the multivariable chain rule:
$$\frac{\partial L}{\partial U[k, j]} = \frac{\partial L}{\partial s[j]} \cdot \frac{\partial s[j]}{\partial U[k, j]} = e[j] \cdot h[k]$$

In matrix outer-product form ($h^T \in \mathbb{R}^{d \times 1}$, $e \in \mathbb{R}^{1 \times V}$):
$$\nabla_U L = h^T e \quad \in \mathbb{R}^{d \times V}$$

### 3.3 Gradient with Respect to Center Embedding $h$
The center representation $h[k]$ influences all $V$ candidate scores $\{s[0], s[1], \dots, s[V-1]\}$.
Applying the multivariable total derivative:
$$\frac{\partial L}{\partial h[k]} = \sum_{j=0}^{V-1} \frac{\partial L}{\partial s[j]} \cdot \frac{\partial s[j]}{\partial h[k]}$$

Since $\frac{\partial s[j]}{\partial h[k]} = U[k, j]$:
$$\nabla_h L[k] = \sum_{j=0}^{V-1} U[k, j] \cdot e[j]$$

In vector-matrix form ($e \in \mathbb{R}^{1 \times V}$, $U^T \in \mathbb{R}^{V \times d}$):
$$\nabla_h L = e U^T \quad \in \mathbb{R}^{1 \times d}$$

> **Critical Implementation Requirement:** $\nabla_h L$ must be computed using the pre-update weights of matrix $U$. Updating $U$ before computing $\nabla_h L$ introduces an erroneous second-order term.

---

## 4. Parameter Update via Stochastic Gradient Descent (SGD)

For a single observed pair $(i, t)$ with learning rate $\eta > 0$:

1. Compute gradients using current weights:
   $$\nabla_U L = h^T e, \quad \nabla_h L = e U^T$$
2. Update output matrix $U$:
   $$U \leftarrow U - \eta \nabla_U L$$
3. Update only the active row in $E$:
   $$E[i, :] \leftarrow E[i, :] - \eta \nabla_h L$$
   $$E[j, :] \leftarrow E[j, :] \quad \text{for all } j \neq i$$

---

## 5. Dataset Loss Evaluation

Dataset loss $J(E, U)$ measures the average cross-entropy across all $N$ training pairs:
$$J(E, U) = \frac{1}{N} \sum_{n=1}^N L_n(E, U)$$
During evaluation, weights $E$ and $U$ remain strictly frozen.

---

## 6. Cosine Similarity Metric

Semantic proximity between two word vectors $a, b \in \mathbb{R}^d$ is measured by the cosine of the angle between them:
$$\text{cosine}(a, b) = \frac{\langle a, b \rangle}{\|a\|_2 \|b\|_2} = \frac{\sum_{k=0}^{d-1} a[k] \cdot b[k]}{\sqrt{\sum_{k=0}^{d-1} a[k]^2} \cdot \sqrt{\sum_{k=0}^{d-1} b[k]^2}}$$
If $\|a\|_2 = 0$ or $\|b\|_2 = 0$, the function returns $0.0$ to prevent division by zero.
