"""Run a short example with the AcedPayloadEnv and RandomAgent."""

import argparse
import numpy as np
import torch
from env import PettingZooPayloadEnv, _make_parallel_pettingzoo_env
from agents import MultiAgentBase
import config


def run_episode(
    env: PettingZooPayloadEnv, agent: MultiAgentBase, render: bool = False
) -> tuple[float, dict]:
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    comm_count = 0
    contact_steps = 0

    while not done:
        # Get comms from agents in env
        comms = env.get_comms()
        actions = agent.select_actions(obs, comms)

        # Step environment
        obs, rewards, done, truncated, info = env.step(actions)
        reward_sum = (
            np.sum(rewards)
            if isinstance(rewards, (np.ndarray, list))
            else float(rewards)
        )
        total_reward += reward_sum

        # Track payload contact
        agents_in_contact = (
            np.linalg.norm(env.agents_pos - env.payload_pos, axis=1)
            < env.contact_radius
        )
        if agents_in_contact.sum() > 0:
            contact_steps += 1

        step += 1
        done = done or truncated

        if render:
            env.render()
        step += 1
    return total_reward, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event", action="store_true", help="Run in event-driven (asynchronous) mode"
    )
    parser.add_argument(
        "--steps", type=int, default=5, help="Number of episodes to run"
    )
    args = parser.parse_args()

    config_manager = config.get_default_configs()["quick_test"]

    env = PettingZooPayloadEnv(config_manager.environment)
    agent = MultiAgentBase(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        n_agents=env.n_agents,
    )

    successes = 0
    for ep in range(args.steps):
        total_reward, info = run_episode(env, agent, render=True)
        print(
            f"Episode {ep+1}: total_reward={total_reward:.3f}, success={info.get('success')}"
        )
        if info.get("success"):
            successes += 1
    print(f"Successes: {successes}/{args.steps}")


if __name__ == "__main__":
    main()
