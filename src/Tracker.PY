
import json
from dataclasses import asdict, dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class TrialMetrics:
    prompt_id: str
    seed: int = 0                    # which evaluation seed this episode belongs to
    pag_guided: bool = False         # explicit, logged per-episode (not just in metadata)
    success: bool = False
    queries_to_success: int = 0
    total_latency: float = 0.0
    causal_overhead_time: float = 0.0
    mutator_tokens: int = 0
    target_tokens: int = 0
    judge_tokens: int = 0
    total_tokens: int = 0
    judge_pattern: str = ""
    causal_vector: Optional[List[float]] = None   # 6D causal state at terminal step


@dataclass
class TrainingMetrics:
    real_training_time: float = 0.0
    total_training_tokens: int = 0
    total_training_calls: int = 0
    total_steps: int = 0
    use_pag_guidance: bool = False   # bake the training condition into the metrics
                                       # object itself so it can never be mislabeled later


class CausalTracker:
    """Collects per-episode TrialMetrics and one TrainingMetrics summary,
    then writes both to disk with a per-seed ASR breakdown printed to stdout."""

    def __init__(self):
        self.episodes: List[dict] = []
        self.training: TrainingMetrics = TrainingMetrics()

    def log_episode(self, m: TrialMetrics) -> None:
        m.total_tokens = int(m.mutator_tokens + m.target_tokens + m.judge_tokens)
        d = asdict(m)
        # causal_vector is stored as a list; serialize to a JSON string so it
        # survives a round-trip through CSV cleanly (lists don't store well in CSV cells)
        if d.get("causal_vector") is not None:
            d["causal_vector"] = json.dumps(d["causal_vector"])
        self.episodes.append(d)

    def log_training(self, m: TrainingMetrics) -> None:
        self.training = m

    def save(self, prefix: str = "run") -> pd.DataFrame:
        if not self.episodes:
            print("CausalTracker: no episodic data points to save.")
            return pd.DataFrame()

        df = pd.DataFrame(self.episodes)
        df["success"] = df["success"].astype(bool)
        df["queries_to_success"] = df["queries_to_success"].astype(int)
        df["pag_guided"] = df["pag_guided"].astype(bool)
        df["seed"] = df["seed"].astype(int)

        episodes_path = f"{prefix}_episodes.csv"
        df.to_csv(episodes_path, index=False)
        print(f"Saved episode-level log: {episodes_path}")

        # Metadata file recording the training condition explicitly, so this
        # run's identity is never ambiguous later.
        meta = {
            "prefix": prefix,
            "trained_with_pag_guidance": self.training.use_pag_guidance,
            "real_training_time_hours": self.training.real_training_time / 3600,
            "total_training_tokens": self.training.total_training_tokens,
            "total_training_calls": self.training.total_training_calls,
            "total_steps": self.training.total_steps,
        }
        meta_path = f"{prefix}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved run metadata: {meta_path}")

        succ = df[df["success"] == True]   # noqa: E712
        fail = df[df["success"] == False]  # noqa: E712

        avg_qct = succ["queries_to_success"].mean() if len(succ) > 0 else 0.0
        avg_succ_lat = succ["total_latency"].mean() if len(succ) > 0 else 0.0
        avg_fail_lat = fail["total_latency"].mean() if len(fail) > 0 else 0.0
        avg_succ_tok = succ["total_tokens"].mean() if len(succ) > 0 else 0.0
        avg_fail_tok = fail["total_tokens"].mean() if len(fail) > 0 else 0.0

        # Per-seed breakdown, since this is the whole point of multi-seed eval
        per_seed = (
            df.groupby("seed")["success"]
            .agg(N="count", Success_Count="sum")
            .assign(ASR_percent=lambda x: 100 * x["Success_Count"] / x["N"])
        )

        print(f"""
==========================================================
  RUN: {prefix}
  Trained with PAG guidance: {self.training.use_pag_guidance}
==========================================================
  Overall success rate (ASR) : {df.success.mean()*100:.2f}%
  Avg QCT (success)          : {avg_qct:.2f} queries
  Causal overhead            : {df.causal_overhead_time.mean()*1000:.2f}ms / episode
----------------------------------------------------------
  RESOURCE METRICS (SUCCESS vs FAILURE)
  Avg latency (S/F)          : {avg_succ_lat:.2f}s / {avg_fail_lat:.2f}s
  Avg tokens  (S/F)          : {avg_succ_tok:.0f} / {avg_fail_tok:.0f}
----------------------------------------------------------
  TRAINING RESOURCE EXPENDITURE
  Real training time         : {self.training.real_training_time/3600:.2f}h
  Training tokens             : {self.training.total_training_tokens:,}
  Training API calls          : {self.training.total_training_calls:,}
==========================================================
  PER-SEED ASR BREAKDOWN
{per_seed.to_string()}
==========================================================
        """)
        return df
