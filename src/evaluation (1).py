"""
Final deterministic evaluation for thesis reporting, and result summarization.

Extracted from untitled64__2_.py. This function appeared twice, verbatim
(lines 1895 and 3292) — consolidated here into a single canonical version.
"""

import time

import pandas as pd

from src.env import CausalRLBreakerEnv


def evaluate_final_model(
    model,
    test_qs,
    test_refs,
    max_steps: int = 6,
    use_pag_guidance: bool = True,
    output_csv: str = None,
) -> pd.DataFrame:
    """Deterministic evaluation for thesis reporting.

    use_pag_guidance=False -> PPO direct policy only
    use_pag_guidance=True  -> PPO + PAG-guided inference
    """
    all_results = []
    start_eval = time.time()

    env = CausalRLBreakerEnv(
        questions=test_qs,
        references=test_refs,
        max_steps=max_steps,
        train_mode=False,
        pag_epsilon=0.1,
        use_pag_guidance=use_pag_guidance,
    )

    for i, q in enumerate(test_qs):
        obs, _ = env.reset(options={"q_idx": i})

        done = False
        steps = 0
        final_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            final_info = info
            steps += 1

        episode_logs = env.episode_logs

        total_tokens = sum(x.get("total_tokens", 0) for x in episode_logs)
        mutator_tokens = sum(x.get("mutator_tokens", 0) for x in episode_logs)
        target_tokens = sum(x.get("target_tokens", 0) for x in episode_logs)
        judge_tokens = sum(x.get("judge_tokens", 0) for x in episode_logs)

        total_latency = sum(x.get("latency_s", 0.0) for x in episode_logs)
        mutator_latency = sum(x.get("mutator_latency_s", 0.0) for x in episode_logs)
        target_latency = sum(x.get("target_latency_s", 0.0) for x in episode_logs)
        judge_latency = sum(x.get("judge_latency_s", 0.0) for x in episode_logs)

        success = bool(final_info.get("success", False))

        all_results.append({
            "question_id": i,
            "success": success,
            "steps": steps,
            "qct": steps if success else 0,
            "ppo_action": final_info.get("ppo_action"),
            "executed_action": final_info.get("executed_action"),
            "pag_guided": final_info.get("pag_guided"),
            "reward": final_info.get("reward"),
            "total_tokens": int(total_tokens),
            "mutator_tokens": int(mutator_tokens),
            "target_tokens": int(target_tokens),
            "judge_tokens": int(judge_tokens),
            "latency_s": float(total_latency),
            "mutator_latency_s": float(mutator_latency),
            "target_latency_s": float(target_latency),
            "judge_latency_s": float(judge_latency),
            "judge_pattern": final_info.get("judge_info", {}).get("pattern", "unknown"),
            "causal_vector": final_info.get("causal_vector"),
            "prompt": final_info.get("prompt"),
        })

        print(f"Eval {i+1}/{len(test_qs)} | success={success} | steps={steps}")

    df = pd.DataFrame(all_results)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Evaluation saved to: {output_csv}")

    elapsed_min = (time.time() - start_eval) / 60
    print(f"Evaluation time: {elapsed_min:.2f} minutes")

    return df


def summarize_final_eval(df: pd.DataFrame, setting_name: str) -> pd.DataFrame:
    success_df = df[df["success"] == True]   # noqa: E712
    fail_df = df[df["success"] == False]     # noqa: E712

    summary = {
        "setting": setting_name,
        "N": len(df),
        "Success_Count": int(df["success"].sum()),
        "Failure_Count": int((df["success"] == False).sum()),  # noqa: E712
        "ASR_percent": float(df["success"].mean() * 100),
        "Avg_steps_all": float(df["steps"].mean()),
        "Avg_QCT_success_only": float(success_df["qct"].mean()) if len(success_df) else 0.0,
        "Avg_tokens_all": float(df["total_tokens"].mean()),
        "Avg_tokens_success": float(success_df["total_tokens"].mean()) if len(success_df) else 0.0,
        "Avg_tokens_failure": float(fail_df["total_tokens"].mean()) if len(fail_df) else 0.0,
        "Avg_latency_all_s": float(df["latency_s"].mean()),
        "Avg_latency_success_s": float(success_df["latency_s"].mean()) if len(success_df) else 0.0,
        "Avg_latency_failure_s": float(fail_df["latency_s"].mean()) if len(fail_df) else 0.0,
    }

    return pd.DataFrame([summary])
