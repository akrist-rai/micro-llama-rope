# Micro-Llama: Rotary Positional Embeddings (RoPE)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains a clean, from-scratch implementation of **Rotary Positional Embeddings (RoPE)** using PyTorch. 

This module is part of my ongoing project to build a Llama-style LLM completely from scratch to understand the systems engineering behind Generative AI.

## The Concept

Standard Transformers use absolute positional embeddings (adding a vector to the input). However, this fails to generalize well to sequence lengths longer than what the model saw during training.

**RoPE** solves this by encoding position as a **rotation** in the complex plane. 

### How it works (The Math)
Instead of adding a vector, we rotate the query ($q$) and key ($k$) vectors by an angle $\theta$ proportional to their position $m$.

$$
f(x, m) = x \cdot e^{im\theta}
$$

This ensures that the attention score between two tokens depends only on their **relative distance** ($m - n$), not their absolute position.

*(Note: Visualization diagram included in the repository documentation)*

## Implementation Details

I implemented this using PyTorch's complex number support for efficiency:

1.  **Precompute Frequencies:** Calculate the $\theta$ values and convert them to complex polar form (`cis`).
2.  **Complex View:** Reshape the input tensor to view the last dimension as pairs of real and imaginary numbers.
3.  **Rotation:** Perform element-wise complex multiplication to apply the rotation.

## Usage

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

print(q_rotated.shape) # Output: torch.Size([2, 10, 4, 64])
