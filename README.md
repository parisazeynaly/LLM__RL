# CausalRLBreaker: Security Testing of LLMs via Causal RL
[![PyPI version](https://badge.fury.io/py/causal-rl-shaping.svg)](https://badge.fury.io/py/causal-rl-shaping)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%8F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/parisaze/Reward)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official codebase for the Master's Thesis: **"Security Testing of Large Language Models via Causal Reinforcement Learning"** (University of Naples Federico II).

##  Abstract
`CausalRLBreaker` resolves the **credit assignment problem** in automated black-box red-teaming by combining Fast Causal Inference (FCI) trajectory mapping with Potential-Based Reward Shaping (PBRS) and $do$-intervention guided action selection.

---

##  Key Results (AdvBench on Llama-3.3-70B)
| Method | ASR (%) | Avg. Tokens/Ep | Latency (s) | Training API Calls |
| :--- | :---: | :---: | :---: | :---: |
| **DAN (Static Prompting)** | 18.27 ± 2.88 | 222.99 | — | — |
| **RLBreaker (Standard PPO)** | 19.33 ± 3.21 | 346.01 | ~7.20 | 20,480 |
| **CausalRLBreaker (Ours)** | **61.54 ± 1.67** | **1735.55** | **3.30** | **11,915 (-41.8%)** |

*  **3.18× ASR improvement** over RL baselines.
*  **61.44% token cost reduction** during training.
*  **Near-halving of cross-seed variance** ($\pm 3.21 \to \pm 1.67$).

---

## Key Methodological Contributions
1. **Causal Observation Vector:** Maps raw prompt mutations into a 6D continuous causal state $s_t \in [0, 1]^6$.
2. **Dense Causal Reward Shaping:** Employs an online SCM potential function $\Phi(s_t)$ grounded in Potential-Based Reward Shaping (PBRS) to provably preserve optimal policy invariance.
3. **Causal-Guided Action Selection:** Re-ranks candidate mutations via estimated $do$-intervention effects $\Delta\Phi(a)$.

---

## 💻 Quick Start

### 1. Installation
```bash
git clone [https://github.com/parisazeynaly/CausalRLBreaker.git](https://github.com/parisazeynaly/CausalRLBreaker.git)
cd CausalRLBreaker
pip install -r requirements.txt
