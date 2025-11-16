"""Run a short example with the AcedPayloadEnv and RandomAgent."""

import argparse
import numpy as np
from env import AcedPayloadEnv
from agents.random_agent import RandomAgent


def run_episode(
    env: AcedPayloadEnv, agent: RandomAgent, render: bool = False
) -> tuple[float, dict]:
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    step = 0
    while not done and step < env.max_steps:
        # readiness mask for event-driven mode: agents with cooldown==0
        obs_arr = obs
        if env.event_driven:
            readiness = np.array(
                [1 if c == 0 else 0 for c in env.cooldowns], dtype=bool
            )
        else:
            readiness = None
        actions = agent.act(obs_arr, readiness_mask=readiness)
        obs, rewards, done, truncated, info = env.step(actions)
        total_reward += rewards[0]
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

    env = AcedPayloadEnv(n_agents=3, event_driven=args.event, cooldown_time=3)
    agent = RandomAgent(n_agents=env.n_agents)

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
