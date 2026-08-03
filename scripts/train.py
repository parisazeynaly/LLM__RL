"""
Train CausalRLBreaker's PPO policy.

Unlike the earlier draft entrypoint, every step below is wired to a real,
importable module — there is no placeholder / simulated logic left.

Usage:
    export GROQ_API_KEY=...
    python scripts/train.py --seed 42 --total_steps 10000 --use_pag --output_prefix run_v1
"""

import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.callbacks import SCMMonitorCallback
from src.causal.scm import save_scm
from src.data import load_advbench_split
from src.env import CausalRLBreakerEnv
from src.llm.clients import PIPELINE_COUNTERS
from src.metrics import CAUSAL_TRACKER, TrainingMetrics
from src.utils import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train CausalRLBreaker's PPO policy.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--total_steps", type=int, default=10000, help="Total PPO training timesteps.")
    parser.add_argument("--max_steps_per_episode", type=int, default=6, help="Max mutation steps per episode.")
    parser.add_argument("--use_pag", action="store_true", help="Enable PAG-guided action selection during training.")
    parser.add_argument("--data_path", type=str, default="data/advbench.csv", help="Path to the AdvBench CSV.")
    parser.add_argument("--checkpoint_dir", type=str, default="ckpt/", help="Directory for PPO checkpoints.")
    parser.add_argument("--output_prefix", type=str, default="run_v1", help="Prefix for saved logs/SCM/model.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Initializing training | seed={args.seed} | PAG guidance={args.use_pag}")

    set_global_seed(args.seed)

    train_qs, test_qs, train_refs, test_refs = load_advbench_split(
        args.data_path, test_size=0.20, random_state=42
    )

    def make_env():
        return CausalRLBreakerEnv(
            questions=train_qs,
            references=train_refs,
            max_steps=args.max_steps_per_episode,
            train_mode=True,
            pag_epsilon=0.2,
            use_pag_guidance=args.use_pag,
        )

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        learning_rate=2e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        normalize_advantage=True,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=3200,
        save_path=args.checkpoint_dir,
        name_prefix=args.output_prefix,
    )
    scm_monitor_cb = SCMMonitorCallback(check_freq=1000)

    start = time.time()
    model.learn(
        total_timesteps=args.total_steps,
        callback=[checkpoint_cb, scm_monitor_cb],
    )
    real_training_time = time.time() - start

    model_path = f"{args.checkpoint_dir}{args.output_prefix}_final.zip"
    model.save(model_path)
    print(f"Model saved -> {model_path}")

    scm_path = f"{args.checkpoint_dir}{args.output_prefix}_scm.npz"
    save_scm(scm_path)

    training_meta = TrainingMetrics(
        real_training_time=real_training_time,
        total_training_tokens=(
            PIPELINE_COUNTERS.total_mutator_tokens
            + PIPELINE_COUNTERS.total_target_tokens
            + PIPELINE_COUNTERS.total_judge_tokens
        ),
        total_training_calls=PIPELINE_COUNTERS.total_api_calls,
        total_steps=args.total_steps,
        use_pag_guidance=args.use_pag,
    )

    CAUSAL_TRACKER.log_training(training_meta)
    CAUSAL_TRACKER.save(prefix=args.output_prefix)

    print(f"Training complete in {real_training_time/3600:.2f}h. Artifacts saved with prefix '{args.output_prefix}'.")


if __name__ == "__main__":
    main()
