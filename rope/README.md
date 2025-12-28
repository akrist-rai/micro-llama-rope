# Micro-Llama: Rotary Positional Embeddings (RoPE)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Implementation_Complete-green?style=for-the-badge)

This repository contains a clean, from-scratch implementation of **Rotary Positional Embeddings (RoPE)** using PyTorch. 

This module is part of my ongoing project to build a Llama-style LLM completely from scratch to understand the systems engineering behind Generative AI.

---

## 💡 The Concept

Standard Transformers use **Absolute Positional Embeddings** (adding a learned vector to the input). However, this fails to generalize to sequence lengths longer than what the model saw during training.

**RoPE** solves this by encoding position as a **rotation** in the complex plane, rather than an addition.

> [!IMPORTANT]
> **The Breakthrough:** Instead of adding a vector, we rotate the query ($q$) and key ($k$) vectors by an angle $\theta$ proportional to their position $m$. This preserves the norm of the vectors while encoding their order.

---

## 📐 The Mathematics

The core idea is to transform a pair of features $(x_1, x_2)$ at position $m$ by rotating them in the 2D plane.

### 1. The Rotation Matrix
For a dimension $d=2$, the function $f(x, m)$ is defined as a geometric rotation:

$$
f(x, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

### 2. Generalization to $d$-dimensions
For high-dimensional embeddings (e.g., $d=4096$), we divide the vector into $d/2$ pairs and rotate each pair with a different frequency $\theta_i$. This creates a block-diagonal rotation matrix:

$$
R_{\Theta, m}^d = \begin{pmatrix} 
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots 
\end{pmatrix}
$$

### 3. Relative Positioning (The "Magic" Part)
The attention score is the dot product of Query ($q$) and Key ($k$). When we apply RoPE, the attention score becomes:

$$
\langle q, k \rangle = \text{Re}\left( \sum_{j=1}^{d/2} q_j k_j^* e^{i(m-n)\theta_j} \right)
$$

Notice that the result depends **only** on $(m-n)$ (the relative distance), not on the absolute values of $m$ or $n$. This is why Llama 2 can extrapolate to longer sequences effectively.

---

## 🖼️ Visualizing the Mechanism

### The Implementation Flow
*This diagram illustrates how the input vectors are split, rotated, and recombined.*

![RoPE Architecture Diagram](./rope_diagram.jpg)


![Geometric Rotation](https://raw.githubusercontent.com/meta-llama/llama/main/assets/rope_visual.png)
*(Placeholder: The rotation in the complex plane allows the model to "sense" distance via angle)*

---

## 💻 Implementation Details

I implemented this using PyTorch's complex number support for maximum efficiency. The naive matrix multiplication is slow, so we use the element-wise polar form:

1.  **Precompute Frequencies:** Calculate $\theta$ values: $\theta_i = 10000^{-2i/d}$.
2.  **Complex View:** Reshape the input tensor `(B, Seq, Heads, Dim)` to pair adjacent elements.
3.  **Apply Rotation:**
    ```python
    # Polar form rotation: x_out = x_complex * e^(i * m * theta)
    x_rotated = x_complex * freqs_cis
    ```

## 🚀 Usage

```python
import torch
from rope import RotaryPositionalEmbeddings

# Initialize RoPE with a head dimension of 64
rope = RotaryPositionalEmbeddings(d_model=64)

# Create a dummy query tensor
# (Batch: 2, Seq: 10, Heads: 4, Dim: 64)
q = torch.randn(2, 10, 4, 64)

# Apply rotation
q_rotated = rope(q)

print(f"Input shape: {q.shape}")       # torch.Size([2, 10, 4, 64])
print(f"Rotated shape: {q_rotated.shape}") # torch.Size([2, 10, 4, 64])
