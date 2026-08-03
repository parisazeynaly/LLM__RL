# Security Testing of Large Language Models via Causal Reinforcement Learning

**Status:** Manuscript finalized · targeting a peer-reviewed security/ML-safety venue
**Author:** Parisa Zeinaliashtiyani · **Supervisor:** Prof. Roberto Pietrantuono
University of Naples Federico II — M.Sc. Thesis in Data Science & Engineering

---

## Overview

This repository contains the implementation and experimental pipeline for **CausalRLBreaker**, a framework that studies whether embedding causal structure into a reinforcement‑learning agent improves LLM security testing over standard RL.

Automated security testing of LLMs increasingly relies on RL agents that learn prompt‑mutation policies through environment interaction (e.g. RLBreaker). These agents face a **credit assignment problem**: when several mutations precede a guardrail bypass, sparse terminal rewards distribute credit equally across all of them, wasting exploration budget on mutations that had no real effect. This project asks whether discovering *which* structural properties of a prompt are causally responsible for a bypass — and using that structure to guide both training and inference — resolves this bottleneck.

## Research questions

- **RQ1** — Is RL‑based mutation more effective than static, dataset‑based prompting for LLM security testing? *(RLBreaker vs. the DAN dataset, Shen et al., arXiv:2308.03825, 2023)*
- **RQ2** — Does incorporating causal knowledge into the RL framework (**Causal RL**) further improve effectiveness and efficiency over standard RL?

## Method summary

1. **Causal discovery via trajectory mapping** — the Fast Causal Inference (FCI) algorithm (via a COAT‑style pipeline) is run on logged RL trajectories to estimate a Partial Ancestral Graph (PAG) over six prompt‑level structural factors.
2. **Dense causal reward shaping** — an online, OLS‑adaptive Structural Causal Model (SCM) turns the PAG into step‑level, potential‑based reward signal, replacing sparse terminal feedback.
3. **Causal action masking** — the learned causal structure constrains the action space at inference time, pruning mutations unrelated to a bypass.

The agent is trained with PPO (Stable‑Baselines3) inside a modular Gymnasium environment that orchestrates a distributed rewriter → target → judge pipeline (Llama‑3.3‑70B as target, Llama‑4‑Scout as judge).

## Results

Evaluated on 104 held‑out AdvBench prompts, 3 random seeds, against Llama‑3.3‑70B:

| Method | Success rate (ASR) | Training‑time API calls |
|---|---|---|
| DAN (static) | 18.27% ± 2.88% | — |
| RLBreaker (RL‑only) | 19.33% ± 3.21% | 20,480 |
| **CausalRLBreaker (ours)** | **61.54% ± 1.67%** | **11,915 (‑41.8%)** |

Zero‑shot transfer to a different model family (Qwen‑3) reaches 45.19% ASR without retraining. Full results, ablations, and discussion are in the manuscript.

## Repository structure

```
Causal RLbreaker/          # core CausalRLBreaker implementation
empirical_evaluation/      # evaluation scripts and result artifacts
RLBreaker.ipynb            # RL-only baseline replication
Factor_proposal.ipynb      # causal factor extraction / proposal notebook
causal_guided_generator_v41.py
coat_input_advbench.csv    # COAT pipeline input (AdvBench-derived)
factor_round2.json         # extracted causal factor activations
fci-round2.png             # estimated PAG (Figure 1 in the manuscript)
annotations_round2_only (1).csv
rlbreaker_light_runs (17).csv
2402.03941v3 (3).pdf       # reference paper (FCI / causal discovery background)
index.html                 # portfolio site (unrelated to the research pipeline — see below)
```

> **Note:** `index.html` is a personal portfolio site hosted from this repo via GitHub Pages and is unrelated to the research code above. It will move to a separate repository in a future cleanup pass.

## Setup

```bash
pip install stable-baselines3 gymnasium causal-learn pandas numpy
```

The pipeline expects API access to the target/judge/mutator models (configured via environment variables — see the notebooks for the exact variable names). Results and logs are written as structured CSV/JSON artifacts under `empirical_evaluation/`.

## Ethics & responsible disclosure

This is defensive AI‑safety evaluation research, not an offensive toolkit. All experiments run against the public AdvBench benchmark under controlled, non‑deployed conditions. No successful bypass prompts, raw model outputs, or exploit templates are published in this repository.

## Citation

```bibtex
@unpublished{zeinaliashtiyani2026causalrlbreaker,
  title  = {Security Testing of Large Language Models via Causal Reinforcement Learning},
  author = {Zeinaliashtiyani, Parisa and Pietrantuono, Roberto},
  year   = {2026},
  note   = {Manuscript finalized; targeting a peer-reviewed security/ML-safety venue}
}
```

## Contact

Parisa Zeinaliashtiyani — [p.zeinaliashtiyani@studenti.unina.it](mailto:p.zeinaliashtiyani@studenti.unina.it) · [GitHub](https://github.com/parisazeynaly) · [LinkedIn](https://www.linkedin.com/in/parisa-zeynaly/)
