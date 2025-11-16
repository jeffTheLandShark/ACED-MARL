import math
import random
from typing import List, Tuple, Dict, Optional
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AcedPayloadEnv(gym.Env):
    """
    A simplified 2D multi-agent payload-transport environment with partial observability.

    - n_agents: number of rovers
    - step-based (synchronous) mode: all agents act every environment step
    - event-driven mode: agents have cooldowns; receiving messages triggers readiness

    Observation per agent (shape: (n_agents, obs_dim)):
    [vx, vy, dist_payload, dist_goal, msg_vx, msg_vy, msg_dist_payload, msg_dist_goal, msg_age, cooldown]

    Action space: MultiDiscrete([6]*n_agents)
     0: NOOP / hold
     1: +x velocity
     2: -x velocity
     3: +y velocity
     4: -y velocity
     5: update (broadcast velocity and beacon readings)

    Reward: progress of payload toward goal; bonus for reaching goal; penalty for losing contact.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        n_agents: int = 5,
        arena_size: float = 20.0,
        goal_radius: float = 5.0,
        goal_border_offset: float = 0.1,
        max_steps: int = 200,
        event_driven: bool = False,
        cooldown_time: int = 0,
        contact_radius: float = 2.0,
        placement_radius: float = 0.9,
        max_agent_speed: float = 5.0,
        comm_dropout_prob: float = 0.0,
        comm_delay_steps: int = 0,
        sensor_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.arena_size = arena_size
        self.goal_radius = goal_radius
        self.goal_border_offset = goal_border_offset
        self.max_steps = max_steps
        self.event_driven = event_driven
        self.cooldown_time = cooldown_time
        self.contact_radius = contact_radius
        self.placement_radius = placement_radius
        self.max_agent_speed = max_agent_speed

        # Robustness factors
        self.comm_dropout_prob = comm_dropout_prob
        self.comm_delay_steps = comm_delay_steps
        self.sensor_dropout_prob = sensor_dropout_prob

        # observation: [vx, vy, dist_payload, dist_goal, msg_vx, msg_vy,
        #               msg_dist_payload, msg_dist_goal, msg_age, cooldown]
        self.obs_dim = 10
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(self.n_agents, self.obs_dim),
            dtype=np.float32,
        )
        # actions per agent: 0..5
        self.action_space = spaces.MultiDiscrete([6] * self.n_agents)

        # internal state
        self.goal = np.zeros(2, dtype=float)
        self.agents_pos = np.zeros((self.n_agents, 2), dtype=float)
        self.agents_vel = np.zeros((self.n_agents, 2), dtype=float)
        self.payload_pos = np.array([arena_size / 2.0, arena_size / 2.0], dtype=float)
        self.payload_vel = np.zeros(2, dtype=float)
        self.step_count = 0

        # communication state: last message (vx, vy, dist_payload, dist_goal) and age
        self.last_messages = [None for _ in range(self.n_agents)]
        self.msg_age = [999 for _ in range(self.n_agents)]

        # message queue for delayed communication
        self.message_queue = deque()

        # event-driven readiness / cooldown per agent
        self.cooldowns = [0 for _ in range(self.n_agents)]

        # tracking metrics
        self.total_contact_time = [0 for _ in range(self.n_agents)]
        self.total_updates_sent = 0

        # random seed
        self.seed()

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        random.seed(seed)

    def _place_goal(self, border_offset: float = 0.1):
        """Place goal randomly on arena border."""
        side = self._rng.integers(0, 4)
        match side:
            case 0:  # left
                self.goal = np.array(
                    [
                        self._rng.uniform(0.0, self.arena_size * border_offset),
                        self._rng.uniform(0.0, self.arena_size),
                    ],
                    dtype=float,
                )
            case 1:  # right
                self.goal = np.array(
                    [
                        self._rng.uniform(
                            self.arena_size * (1 - border_offset), self.arena_size
                        ),
                        self._rng.uniform(0.0, self.arena_size),
                    ],
                    dtype=float,
                )
            case 2:  # bottom
                self.goal = np.array(
                    [
                        self._rng.uniform(0.0, self.arena_size),
                        self._rng.uniform(0.0, self.arena_size * border_offset),
                    ],
                    dtype=float,
                )
            case 3:  # top
                self.goal = np.array(
                    [
                        self._rng.uniform(0.0, self.arena_size),
                        self._rng.uniform(
                            self.arena_size * (1 - border_offset), self.arena_size
                        ),
                    ],
                    dtype=float,
                )
            case _:
                self.goal = np.array(
                    [self.arena_size, self.arena_size / 2.0], dtype=float
                )

    def _place_payload(self, offset: float = 0.2):
        """Place payload near center with agents in contact range."""
        self.payload_pos = np.array(
            [
                self.arena_size / 2.0
                + self._rng.uniform(
                    -self.arena_size * offset, self.arena_size * offset
                ),
                self.arena_size / 2.0
                + self._rng.uniform(
                    -self.arena_size * offset, self.arena_size * offset
                ),
            ],
            dtype=float,
        )
        self.payload_vel = np.zeros(2)

        # Initialize agents within contact radius
        for i in range(self.n_agents):
            angle = self._rng.uniform(0.0, 2.0 * math.pi)
            r = self._rng.uniform(0.0, self.contact_radius * self.placement_radius)
            offset_vec = np.array([math.cos(angle) * r, math.sin(angle) * r])
            pos = self.payload_pos + offset_vec
            self.agents_pos[i] = np.clip(pos, 0.0, self.arena_size)
            self.agents_vel[i] = np.zeros(2)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)

        self._place_goal()
        self._place_payload()
        self.step_count = 0
        self.last_messages = [None for _ in range(self.n_agents)]
        self.msg_age = [999 for _ in range(self.n_agents)]
        self.cooldowns = [0 for _ in range(self.n_agents)]
        self.message_queue.clear()
        self.total_contact_time = [0 for _ in range(self.n_agents)]
        self.total_updates_sent = 0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        """
        Observation per agent (partial observability):
        [vx, vy, dist_payload, dist_goal, msg_vx, msg_vy,
         msg_dist_payload, msg_dist_goal, msg_age, cooldown]
        """
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for i in range(self.n_agents):
            vx, vy = self.agents_vel[i]

            # Scalar distances only (beacons don't provide direction)
            dist_payload = np.linalg.norm(self.payload_pos - self.agents_pos[i])
            dist_goal = np.linalg.norm(self.goal - self.agents_pos[i])

            # Apply sensor dropout
            if self._rng.random() < self.sensor_dropout_prob:
                dist_payload = 0.0
                dist_goal = 0.0

            msg = self.last_messages[i]
            if msg is None:
                msg_vx, msg_vy, msg_dist_payload, msg_dist_goal = 0.0, 0.0, 0.0, 0.0
                age = 999.0
            else:
                msg_vx, msg_vy, msg_dist_payload, msg_dist_goal = msg
                age = float(self.msg_age[i])

            cd = float(self.cooldowns[i])
            obs[i] = np.array(
                [
                    vx,
                    vy,
                    dist_payload,
                    dist_goal,
                    msg_vx,
                    msg_vy,
                    msg_dist_payload,
                    msg_dist_goal,
                    age,
                    cd,
                ],
                dtype=np.float32,
            )
        return obs

    def step(self, actions) -> Tuple[np.ndarray, list[float], bool, bool, Dict]:
        """Execute one environment step."""
        self.step_count += 1

        actions = np.asarray(actions, dtype=int)
        if actions.shape[0] != self.n_agents:
            raise ValueError("actions must be length n_agents")

        # Process delayed messages from queue
        current_messages = []
        while self.message_queue and self.message_queue[0][0] <= self.step_count:
            _, sender, msg = self.message_queue.popleft()
            current_messages.append((sender, msg))

        # Apply actions
        broadcasts: List[Tuple[int, Tuple[float, float, float, float]]] = []
        for i in range(self.n_agents):
            a = int(actions[i])
            if self.event_driven and self.cooldowns[i] > 0:
                a = 0  # NOOP if on cooldown

            if a == 0:
                pass  # hold
            elif a == 1:
                self.agents_vel[i][0] += 1.0
            elif a == 2:
                self.agents_vel[i][0] -= 1.0
            elif a == 3:
                self.agents_vel[i][1] += 1.0
            elif a == 4:
                self.agents_vel[i][1] -= 1.0
            elif a == 5:
                # Broadcast: velocity and beacon readings
                dist_payload = np.linalg.norm(self.payload_pos - self.agents_pos[i])
                dist_goal = np.linalg.norm(self.goal - self.agents_pos[i])

                msg = (
                    float(self.agents_vel[i][0]),
                    float(self.agents_vel[i][1]),
                    float(dist_payload),
                    float(dist_goal),
                )

                # Apply communication dropout
                if self._rng.random() >= self.comm_dropout_prob:
                    broadcasts.append((i, msg))
                    self.total_updates_sent += 1

                # Broadcasting agent enters cooldown
                self.cooldowns[i] = self.cooldown_time

        # Update message ages
        for i in range(self.n_agents):
            if self.last_messages[i] is not None:
                self.msg_age[i] += 1

        # Deliver broadcasts (with optional delay)
        all_broadcasts = current_messages + broadcasts
        for sender, msg in all_broadcasts:
            for i in range(self.n_agents):
                if i == sender:
                    continue

                if self.comm_delay_steps > 0:
                    # Queue message for future delivery
                    delivery_step = self.step_count + self.comm_delay_steps
                    self.message_queue.append((delivery_step, sender, msg))
                else:
                    # Immediate delivery
                    self.last_messages[i] = msg
                    self.msg_age[i] = 0
                    if self.event_driven:
                        self.cooldowns[i] = 0

        # Physics: clamp velocities BEFORE updating positions
        for i in range(self.n_agents):
            speed = np.linalg.norm(self.agents_vel[i])
            if speed > self.max_agent_speed:
                self.agents_vel[i] = (self.agents_vel[i] / speed) * self.max_agent_speed

        # Update agent positions
        for i in range(self.n_agents):
            self.agents_pos[i] += self.agents_vel[i] * 0.1
            self.agents_pos[i] = np.clip(self.agents_pos[i], 0.0, self.arena_size)

        # Payload physics: only agents in contact affect it
        distances = np.linalg.norm(self.agents_pos - self.payload_pos, axis=1)
        agents_in_contact = distances <= self.contact_radius

        # Track contact time
        for i in range(self.n_agents):
            if agents_in_contact[i]:
                self.total_contact_time[i] += 1

        dist_to_goal_before = np.linalg.norm(self.goal - self.payload_pos)

        if agents_in_contact.sum() > 0:
            self.payload_vel = np.mean(self.agents_vel[agents_in_contact], axis=0)
        else:
            self.payload_vel *= 0.9  # decay if no contact

        self.payload_pos += self.payload_vel
        self.payload_pos = np.clip(self.payload_pos, 0.0, self.arena_size)

        # Decay cooldowns
        for i in range(self.n_agents):
            if self.cooldowns[i] > 0:
                self.cooldowns[i] = max(0, self.cooldowns[i] - 1)

        # Compute reward
        dist_to_goal_after = np.linalg.norm(self.goal - self.payload_pos)
        base_reward = dist_to_goal_before - dist_to_goal_after
        rewards = np.full(self.n_agents, base_reward)

        # Penalty for lost contact
        rewards -= (1.0 - agents_in_contact) * 5.0
        contact_fraction = agents_in_contact.sum() / self.n_agents

        # Check termination conditions
        done = False
        success = False

        # Success: payload reaches goal
        if dist_to_goal_after < self.goal_radius:
            rewards += 100.0
            done = True
            success = True

        # Failure: too many agents lost contact
        if contact_fraction < 0.5:
            rewards -= 10.0
            done = True
            success = False

        # Max steps
        if self.step_count >= self.max_steps:
            done = True

        obs = self._get_obs()

        # Compute metrics
        contact_fractions = [
            self.total_contact_time[i] / self.step_count for i in range(self.n_agents)
        ]

        info = {
            "success": success,
            "payload_pos": self.payload_pos.copy(),
            "agents_pos": self.agents_pos.copy(),
            "contact_fraction": contact_fraction,
            "avg_contact_fraction": np.mean(contact_fractions),
            "num_updates": self.total_updates_sent,
            "dist_to_goal": dist_to_goal_after,
        }

        return obs, rewards, done, False, info

    def render(self, mode: str = "human"):
        """ASCII render of environment state."""
        grid = np.full((int(self.arena_size), int(self.arena_size)), ".", dtype=str)
        px, py = int(self.payload_pos[0]), int(self.payload_pos[1])
        gx_f, gy_f = float(self.goal[0]), float(self.goal[1])
        gx, gy = int(gx_f), int(gy_f)

        # Mark goal region
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if math.hypot(float(x) - gx_f, float(y) - gy_f) <= self.goal_radius:
                    grid[y, x] = "g"

        # Mark payload contact region
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if math.hypot(float(x) - px, float(y) - py) <= self.contact_radius:
                    grid[y, x] = "p"

        # Mark centers
        grid[gy, gx] = "G"
        grid[py, px] = "P"

        # Mark agents
        for i in range(self.n_agents):
            x, y = int(self.agents_pos[i][0]), int(self.agents_pos[i][1])
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                grid[y, x] = str(i % 10)

        out = "\n".join("".join(row) for row in grid[::-1])
        print(out)
        print(f"Distance to goal: {np.linalg.norm(self.payload_pos - self.goal):.2f}")
        print(f"Step: {self.step_count}, Updates sent: {self.total_updates_sent}\n")
