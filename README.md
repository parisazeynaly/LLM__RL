# CausalRLBreaker: Security Testing of LLMs via Causal RL

### Causal Reinforcement Learning for Automated Security Testing of Large Language Models

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/causal-rl-shaping.svg)](https://badge.fury.io/py/causal-rl-shaping)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](YOUR_HUGGING_FACE_SPACE_URL)
[![Research Project](https://img.shields.io/badge/Project-Research%20Code-blue)](https://github.com/parisazeynaly/Security-Testing-of-Large-Language-Models-via-Reinforcement-Learning)

Official research implementation accompanying the M.Sc. thesis:

> **Security Testing of Large Language Models via Causal Reinforcement Learning**

University of Naples Federico II (UniNa)

---

## Overview

Automated security testing of large language models (LLMs) can be formulated
as a sequential search problem in which an agent iteratively transforms
prompts to discover effective adversarial strategies.

A central challenge is **credit assignment**: black-box feedback from the
target model provides limited information about which intermediate
transformations contributed to a successful outcome.

**CausalRLBreaker** investigates whether causal information can be incorporated
into reinforcement learning to provide more informative state representations,
reward signals, and action-selection mechanisms for automated LLM security
testing.

The framework combines:

- **Fast Causal Inference (FCI)** for estimating causal structure from
  interaction trajectories;
- **Structural Causal Models (SCMs)** for representing causal relationships;
- **Potential-Based Reward Shaping (PBRS)** for incorporating causal information
  into the reward signal; and
- **$do$-intervention-guided action selection** for prioritizing candidate
  prompt transformations.

---

## Research Question

> **Can causal information improve the effectiveness and efficiency of
> reinforcement-learning-based black-box security testing of large language
> models?**

The project investigates this question by integrating causal discovery and
structural causal reasoning directly into an RL-based red-teaming pipeline.

---
## Key Contributions

### 1. Causal Observation Vector

CausalRLBreaker maps raw prompt mutations and trajectory information into a
**6-dimensional continuous causal state representation**:

\[
s_t \in [0,1]^6.
\]

This representation provides the reinforcement learning agent with structured
information beyond the raw black-box response signal.

---

### 2. Dense Causal Reward Shaping

The framework introduces an online SCM-derived potential function

\[
\Phi(s_t)
\]

and incorporates it into **Potential-Based Reward Shaping (PBRS)**.

The objective is to provide denser causal feedback during exploration while
retaining the underlying reinforcement learning objective.

---
---

## 💻 Quick Start

### 1. Installation
```bash
git clone [https://github.com/parisazeynaly/CausalRLBreaker.git](https://github.com/parisazeynaly/CausalRLBreaker.git)
cd CausalRLBreaker
pip install -r requirements.txt
### 3. Causal-Guided Action Selection

Candidate prompt transformations are re-ranked using estimated causal
intervention effects:

\[
\Delta \Phi(a).
\]

This mechanism provides causal guidance for exploration and complements the
underlying PPO policy.

---
## Method

The high-level CausalRLBreaker pipeline is:

```text
                 Initial Prompt
                       |
                       v
              Prompt Transformation
                       |
                       v
              Causal State Vector
                       |
                       v
                FCI / Causal Graph
                       |
                       v
                 Structural Causal
                      Model
                       |
              +--------+--------+
              |                 |
              v                 v
       Causal Potential   Intervention Effects
              |                 |
              +--------+--------+
                       |
                       v
                  PPO Policy
                       |
                       v
              Candidate Action
                       |
                       v
                  Target LLM
                       |
                       v
              Security Evaluation
                       |
                       v
                    Reward
                       |
                       v
                  Next State
              
