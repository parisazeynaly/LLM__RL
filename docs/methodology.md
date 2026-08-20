# Methodology

## 1. Overview

CausalRLBreaker investigates whether causal information can improve
reinforcement-learning-based security testing of large language models.

The framework treats automated LLM red-teaming as a sequential decision
problem in which an agent iteratively modifies prompts in order to discover
transformations that increase the probability of eliciting an undesirable
or policy-violating model response.

The central idea is to incorporate causal information into multiple stages
of the reinforcement learning pipeline rather than relying exclusively on
black-box reward feedback.

---

## 2. Problem Formulation

Let the prompt state at step \(t\) be represented by \(s_t\), and let
\(a_t\) denote a prompt transformation or mutation.

The agent interacts with a target language model and receives a response
that is evaluated by a judge or security criterion.

The resulting transition can be represented as:

\[
s_t \xrightarrow{a_t} s_{t+1}
\]

with reward:

\[
r_t = R(s_t, a_t, s_{t+1}).
\]

The objective is to learn a policy

\[
\pi(a_t \mid s_t)
\]

that efficiently explores the prompt space and maximizes the probability
of successful security-testing outcomes.

---

## 3. Causal State Representation

CausalRLBreaker augments the reinforcement learning state with information
derived from causal relationships among prompt and trajectory features.

The implementation uses causal discovery to estimate a causal structure
from observed data.

The resulting representation is used as part of the agent's observation
space.

The exact feature construction and preprocessing are implemented in the
corresponding source modules.

---

## 4. Causal Discovery

The framework uses constraint-based causal discovery to estimate a
Partial Ancestral Graph (PAG) over the selected variables.

The estimated graph is used as an approximation of the causal structure
available to the agent.

Because the setting is observational and black-box, the discovered graph
should be interpreted as an estimated causal structure under the assumptions
of the underlying causal discovery procedure rather than as ground-truth
causality.

---

## 5. Structural Causal Modeling

The estimated causal structure is represented using a Structural Causal
Model (SCM).

The SCM provides a mechanism for reasoning about changes to variables under
interventions.

For a variable \(X\), an intervention can be conceptually represented as:

\[
do(X=x).
\]

The framework uses estimated intervention effects to inform the
reinforcement learning process.

---

## 6. Causal Reward Shaping

Causal information is incorporated into the reward signal using a
potential-based reward shaping formulation.

Let

\[
\Phi(s)
\]

denote the causal potential associated with state \(s\).

The shaped reward can be written as:

\[
r'_t =
r_t +
\gamma \Phi(s_{t+1})
-
\Phi(s_t).
\]

The purpose of the additional signal is to provide denser feedback related
to the estimated causal structure while retaining the underlying task
objective.

The implementation details and exact potential definition are provided in
the source code and experimental configuration.

---

## 7. Causal-Guided Action Selection

In addition to reward shaping, causal intervention estimates are used to
guide the selection or prioritization of candidate prompt transformations.

The intended effect is to bias exploration toward actions with greater
estimated causal relevance to the target objective.

This mechanism complements the policy learned by PPO rather than replacing
the reinforcement learning algorithm.

---

## 8. Reinforcement Learning

The current implementation uses Proximal Policy Optimization (PPO) as the
reinforcement learning algorithm.

At each step, the policy selects or prioritizes a prompt transformation,
the transformed prompt is evaluated against the target model, and the
resulting feedback is used to update the policy.

The main components are therefore:

1. State construction
2. Candidate action generation
3. Causal guidance
4. Target-model evaluation
5. Reward computation
6. PPO policy update

---

## 9. End-to-End Workflow

The high-level workflow is:

```text
Initial Prompt
      |
      v
Prompt Mutation
      |
      v
Causal State Representation
      |
      v
Causal Structure / SCM
      |
      +----------------------+
      |                      |
      v                      v
Causal Reward          Intervention Effects
      |                      |
      +----------+-----------+
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
             Evaluation
                 |
                 v
               Reward
                 |
                 +------> Next State
