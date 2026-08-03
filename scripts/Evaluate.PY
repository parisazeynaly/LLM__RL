"""
Evaluate a trained CausalRLBreaker PPO policy on the held-out AdvBench test
split, across multiple seeds, matching the thesis's evaluation protocol
(Section 4.4-4.5).

Usage:
    export GROQ_API_KEY=...
    python scripts/evaluate.py --model_path ckpt/run_v1_final.zip \
        --scm_path ckpt/run_v1_scm.npz --seeds 42 100 2026
"""

import argparse

import pandas as pd
from stable_baselines3 import PPO

from src.causal.scm import SCM, load_scm
from src.data import load_advbench_split
from src.evaluation import evaluate_final_model, summarize_final_eval
from src.utils import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CausalRLBreaker model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PPO .zip checkpoint.")
    parser.add_argument("--scm_path", type=str, required=True, help="Path to the saved SCM .npz file.")
    parser.add_argument("--data_path", type=str, default="data/advbench.csv", help="Path to the AdvBench CSV.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026], help="Evaluation seeds.")
    parser.add_argument("--max_steps", type=int, default=6, help="Max mutation steps per episode.")
    parser.add_argument("--use_pag", action="store_true", help="Enable PAG-guided inference for this eval run.")
    parser.add_argument("--output_dir", type=str, default="empirical_evaluation/", help="Where to write result CSVs.")
    return parser.parse_args()


def main():
    args = parse_args()

    model = PPO.load(args.model_path)
    load_scm(args.scm_path)
    SCM.set_eval_mode()
    print("Loaded model and SCM. SCM state:")
    print(SCM.summary())

    # Test split is derived the same way for every seed run — held out once,
    # not re-split per seed, to avoid leakage across the evaluation seeds.
    _, test_qs, _, test_refs = load_advbench_split(args.data_path, test_size=0.20, random_state=42)

    all_summaries = []

    for seed in args.seeds:
        set_global_seed(seed)
        print(f"\n=== Evaluating seed {seed} ===")

        df = evaluate_final_model(
            model=model,
            test_qs=test_qs,
            test_refs=test_refs,
            max_steps=args.max_steps,
            use_pag_guidance=args.use_pag,
            output_csv=f"{args.output_dir}seed_{seed}_episodes.csv",
        )

        summary = summarize_final_eval(df, setting_name=f"seed_{seed}")
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_path = f"{args.output_dir}multi_seed_summary.csv"
    combined.to_csv(combined_path, index=False)

    print(f"\nAll seeds evaluated. Combined summary -> {combined_path}")
    print(combined.to_string())
    print(f"\nMean ASR across seeds: {combined['ASR_percent'].mean():.2f}% (std: {combined['ASR_percent'].std():.2f})")


if __name__ == "__main__":
    main()
