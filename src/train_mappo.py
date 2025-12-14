"""
Simple MAPPO training script using RLlib.

Usage:
    python train_mappo.py --preset quick_test
    python train_mappo.py --config path/to/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env

# RLlib wrapper for PettingZoo environments
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

import config
from env import PettingZooPayloadEnv, register_and_get_env_name


class PayloadMetricsCallbacks(DefaultCallbacks):
    """Collect env info metrics (success, contact, distance)."""

    def on_episode_end(self, *, episode, **kwargs):  # type: ignore[override]
        infos = []
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info:
                infos.append(info)
        if not infos:
            return

        def _avg(key):
            vals = [i[key] for i in infos if key in i]
            return float(sum(vals) / len(vals)) if vals else None

        metrics = {
            "contact_fraction": _avg("contact_fraction"),
            "avg_contact_fraction": _avg("avg_contact_fraction"),
            "success_rate": _avg("success"),
            "dist_to_goal": _avg("dist_to_goal"),
        }
        for k, v in metrics.items():
            if v is not None:
                episode.custom_metrics[k] = v


def create_mappo_config(exp_config: config.ExperimentConfig):
    """Create RLlib PPO configuration for MAPPO training."""

    # Register environment and use the registered name (string) for RLlib
    env_name = register_and_get_env_name(exp_config.environment.__dict__)

    # Create a sample environment to get spaces via PettingZoo wrapper
    sample_env = PettingZooEnv(PettingZooPayloadEnv(exp_config.environment))
    
    # Extract individual agent spaces from the Dict spaces
    # The observation_space and action_space are Dict spaces with per-agent keys
    first_agent = sample_env.agents[0]
    obs_space = sample_env.observation_space[first_agent]
    act_space = sample_env.action_space[first_agent]

    # Build PPO config (modernized API for Ray >= 2.7)
    ppo_config = (
        PPOConfig()
        # Use the registered env name (string). RLlib will create the env via registry.
        .environment(env=env_name)
        .framework("torch")
        # Use legacy API stack to avoid new RLModule/Catalog requirements for Dict spaces
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=exp_config.training.num_workers,
            rollout_fragment_length=exp_config.training.rollout_fragment_length,
        )
        .training(
            train_batch_size=exp_config.training.train_batch_size,
            minibatch_size=exp_config.training.sgd_minibatch_size,
            num_epochs=getattr(
                exp_config.training, "num_epochs", exp_config.training.num_sgd_iter
            ),
            lr=exp_config.training.lr,
            gamma=exp_config.training.gamma,
            lambda_=exp_config.training.lambda_,
            clip_param=exp_config.training.clip_param,
            vf_loss_coeff=exp_config.training.vf_loss_coeff,
            entropy_coeff=exp_config.training.entropy_coeff,
        )
        .resources(
            num_gpus=exp_config.training.num_gpus,
        )
        # Use a single shared policy for PettingZoo multi-agent envs
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .callbacks(PayloadMetricsCallbacks)
        .debugging(
            log_level="WARN",
        )
    )

    return ppo_config


def main():
    parser = argparse.ArgumentParser(
        description="Train MAPPO on AcedPayload environment"
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Name of preset config (e.g., 'quick_test')",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run Ray in local mode for debugging",
    )

    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Whether to NOT use tune.Tuner(), but rather a simple for-loop calling "
        "`algo.train()` repeatedly until one of the stop criteria is met.",
    )
    parser.add_argument(
        "--old-api-stack",
        action="store_true",
        help="Run this script on the old API stack of RLlib.",
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=None,
        help="The number of (remote) EnvRunners to use for the experiment.",
    )

    args = parser.parse_args()

    # Load configuration
    if args.preset:
        configs = config.get_default_configs()
        if args.preset not in configs:
            raise ValueError(
                f"Unknown preset: {args.preset}. Available: {list(configs.keys())}"
            )
        exp_config = configs[args.preset]
    elif args.config:
        exp_config = config.ConfigManager.load_yaml(args.config)
    else:
        print("No config specified, using default quick_test preset")
        exp_config = config.get_default_configs()["quick_test"]

    print(f"\n{'='*60}")
    print(f"Training Configuration: {exp_config.name}")
    print(f"{'='*60}")
    print(f"Environment: {exp_config.environment.n_agents} agents")
    print(f"Agent Type: {exp_config.agent.agent_type}")
    print(f"Use ATOC: {exp_config.agent.use_atoc}")
    print(f"Output Dir: {exp_config.output_dir}")
    print(f"{'='*60}\n")

    # Initialize Ray
    ray.init(local_mode=args.local_mode, ignore_reinit_error=True)

    # Create output directory and ensure Ray gets a proper URI
    output_dir = Path(exp_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_uri = f"file://{output_dir.resolve()}"

    # Save config
    config.ConfigManager.save_yaml(exp_config, str(output_dir / "config.yaml"))

    # Create MAPPO configuration
    ppo_config = create_mappo_config(exp_config)

    # Run training
    stop_criteria = {
        "num_env_steps_sampled_lifetime": exp_config.training.total_timesteps,
    }

    checkpoint_config = tune.CheckpointConfig(
        checkpoint_frequency=exp_config.checkpoint.checkpoint_frequency,
        checkpoint_at_end=exp_config.checkpoint.checkpoint_at_end,
        num_to_keep=exp_config.checkpoint.keep_checkpoints_num,
    )

    print("Starting training...")
    results = tune.Tuner(
        "PPO",
        param_space=ppo_config.to_dict(),
        run_config=tune.RunConfig(
            name=exp_config.name,
            stop=stop_criteria,
            checkpoint_config=checkpoint_config,
            storage_path=storage_uri,
            verbose=1 if exp_config.verbose else 0,
        ),
    ).fit()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Get best result
    try:
        best_result = results.get_best_result(
            metric="env_runners/episode_return_mean", mode="max"
        )
        print(f"\nBest checkpoint: {best_result.checkpoint}")
        print(
            f"Best reward: {best_result.metrics.get('env_runners/episode_return_mean', 0.0):.2f}"
        )
    except Exception as e:
        print(f"\nCould not retrieve best result: {e}")
        print("Training results saved to:", output_dir)

    ray.shutdown()


if __name__ == "__main__":
    main()
