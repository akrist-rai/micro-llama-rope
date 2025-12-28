# Micro-Llama: Building State-of-the-Art LLMs from Scratch

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch&logoColor=white)



## 🚀 The Mission
This repository is a first-principles implementation of a modern Large Language Model (LLM), mirroring the architecture of Llama 2 / Mistral. 

**I am building this project because I found that while there are many tutorials on basic Transformers, there was no single resource containing the implementation of State-of-the-Art (SOTA) features like RoPE, RMSNorm, SwiGLU, and FlashAttention in one cohesive codebase.**

My goal is to bridge the gap between "toy examples" and "production-grade engineering" by implementing every component manually in PyTorch/C++.

---

## 📂 Project Architecture
This repository is organized as a modular library. Each key component of the LLM architecture resides in its own module with dedicated documentation and testing.

| Component | Status | Description |
| :--- | :--- | :--- |
| **[RoPE](./rope)** |  Complete | Rotary Positional Embeddings (Complex number rotation for relative positioning) |
| **RMSNorm** | incomplete | Root Mean Square Normalization for training stability |
| **SwiGLU**  | incomplete | The activation function used in Llama 2 / PaLM |
| **DSA** | incomplete | Grouped/Multi-Query Attention for memory efficiency |
| **Tokenizer** | incomplete | BPE Tokenizer training and inference |
| **Flash attention**| incomplete | optemize kv catche compute method


---

## 🧠 Technical Deep Dives
*Click on the folder links above for detailed mathematical explanations and implementation details of each component.*

### Current Highlight: Rotary Positional Embeddings (RoPE)
Located in `rope/`, this module implements the relative positional encoding used by Llama 2.
- **Math:** Uses complex polar form to rotate query/key vectors.
- **Efficiency:** Pre-computes frequency `cis` values to speed up the forward pass.
- **Visual:** Includes diagrams explaining the rotation mechanics.

---

