"""Run a short example with the AcedPayloadEnv and RandomAgent."""

import argparse
import numpy as np
import torch
from env import PettingZooPayloadEnv
from agents import MultiAgentBase
import config


def run_episode(
    env: PettingZooPayloadEnv, agents: dict[str, MultiAgentBase], render: bool = False
) -> tuple[float, dict]:
    obs_dict, infos = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    agent_ids = list(agents.keys())
    while not done:
        actions = {
            agent_id: agents[agent_id].select_action(obs_dict[agent_id])
            for agent_id in agent_ids
        }
        obs_dict, rewards, dones, trunc, infos = env.step(actions)
        total_reward += sum(rewards.values())
        done = any(dones.values()) or any(trunc.values())

        if render:
            env.render()
        step += 1

    # Return info from first agent (they all share the same base info)
    return total_reward, infos


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
    agents = {}
    for agent_id in env.agents:
        agents[agent_id] = MultiAgentBase(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
        )

    successes = 0
    for ep in range(args.steps):
        total_reward, infos = run_episode(env, agents, render=True)
        print(
            f"Episode {ep+1}: total_reward={total_reward:.3f}, success={info.get('success')}"
        )
        if any(infos[i].get("success", False) for i in infos):
            successes += 1
    print(f"Successes: {successes}/{args.steps}")


if __name__ == "__main__":
    main()
