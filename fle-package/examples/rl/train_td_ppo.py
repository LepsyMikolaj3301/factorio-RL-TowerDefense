"""Train a Tower Defense agent using PPO (Stable-Baselines3).

Usage:
    # Start containers first:
    fle cluster --num-instances 4 --scenario tower_defense

    # Then run training:
    python examples/rl/train_td_ppo.py --num-envs 4 --total-timesteps 500000
"""

import argparse
import os

import gymnasium


def main():
    parser = argparse.ArgumentParser(description="Train Tower Defense PPO agent")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--save-path", type=str, default=None, help="Path to prebuilt save file")
    parser.add_argument("--log-dir", type=str, default="./td_ppo_logs", help="Tensorboard log directory")
    parser.add_argument("--model-path", type=str, default="./td_ppo_model", help="Path to save trained model")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("stable-baselines3 is required. Install with: pip install 'fle[rl]'")
        return

    from fle.env.gym_env.td_vector import make_td_vector_env
    from fle.env.gym_env.action_mask import ActionMaskWrapper
    from fle.env.gym_env.registry import make_td_env

    if args.num_envs > 1:
        env = make_td_vector_env(
            num_envs=args.num_envs,
            save_path=args.save_path,
        )
    else:
        env = make_td_env(save_path=args.save_path)
        env = ActionMaskWrapper(env)

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        verbose=1,
        tensorboard_log=args.log_dir,
    )

    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps)

    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

    env.close()


if __name__ == "__main__":
    main()
