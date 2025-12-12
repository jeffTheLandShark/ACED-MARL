import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from torch import Tensor
from config import AgentConfig


class AttentionUnit(nn.Module):
    """
    Attention unit for ATOC that decides when to communicate and how to integrate messages.
    """

    def __init__(self, hidden_dim: int, thought_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.thought_dim = thought_dim

        # Gate network: decides probability of initiating communication
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Query and key networks for attention mechanism
        self.query_net = nn.Linear(hidden_dim, thought_dim)
        self.key_net = nn.Linear(thought_dim, thought_dim)
        self.value_net = nn.Linear(thought_dim, thought_dim)

    def forward(
        self,
        hidden_state: torch.Tensor,
        thoughts: Optional[torch.Tensor] = None,
        return_gate_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_state: (batch_size, hidden_dim) - agent's internal representation
            thoughts: (batch_size, n_other_agents, thought_dim) - received thought vectors
            return_gate_only: if True, only compute communication gate

        Returns:
            gate_prob: (batch_size, 1) - probability of initiating communication
            integrated_thought: (batch_size, thought_dim) - attention-weighted message integration
            attention_weights: (batch_size, n_other_agents) - attention over received messages
        """
        # Communication gate: should this agent broadcast?
        gate_prob = self.gate_net(hidden_state)

        if return_gate_only or thoughts is None:
            return gate_prob, None, None

        # Attention over received thoughts
        batch_size = hidden_state.shape[0]
        n_messages = thoughts.shape[1] if len(thoughts.shape) == 3 else 1

        # Query from own hidden state
        query = self.query_net(hidden_state)  # (batch, thought_dim)

        # Keys and values from received thoughts
        if len(thoughts.shape) == 2:
            thoughts = thoughts.unsqueeze(1)  # (batch, 1, thought_dim)

        keys = self.key_net(thoughts)  # (batch, n_messages, thought_dim)
        values = self.value_net(thoughts)  # (batch, n_messages, thought_dim)

        # Scaled dot-product attention
        scores = torch.bmm(
            query.unsqueeze(1),  # (batch, 1, thought_dim)
            keys.transpose(1, 2),  # (batch, thought_dim, n_messages)
        ) / np.sqrt(
            self.thought_dim
        )  # (batch, 1, n_messages)

        attention_weights = F.softmax(scores, dim=-1)  # (batch, 1, n_messages)

        # Weighted sum of values
        integrated_thought = torch.bmm(
            attention_weights,  # (batch, 1, n_messages)
            values,  # (batch, n_messages, thought_dim)
        ).squeeze(
            1
        )  # (batch, thought_dim)

        return gate_prob, integrated_thought, attention_weights.squeeze(1)


class MultiAgentBase(nn.Module):
    """
    Base class for multi-agent RL with optional ATOC communication.

    This class can be extended to implement MAPPO, MAT, or other MARL algorithms.
    It handles the ATOC communication logic and provides hooks for different
    policy architectures.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        obs_dim: int = 10,
        action_dim: int = 6,
        n_agents: int = 3,
        hidden_dim: int = 128,
        thought_dim: int = 64,
        use_atoc: bool = False,
        comm_penalty: float = 0.01,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.thought_dim = thought_dim
        self.use_atoc = use_atoc
        self.comm_penalty = comm_penalty
        if config is not None:
            self.__dict__.update(config.__dict__)

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ATOC attention unit (optional)
        if use_atoc:
            self.attention_unit = AttentionUnit(hidden_dim, thought_dim)
            # Thought generator: creates message content from hidden state
            self.thought_generator = nn.Sequential(
                nn.Linear(hidden_dim, thought_dim), nn.Tanh()
            )
            # Fusion layer: combines own hidden state with integrated messages
            self.fusion_layer = nn.Linear(hidden_dim + thought_dim, hidden_dim)

        # Action head (to be used by subclasses)
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value head (for actor-critic methods)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        # Centralized critic for CTDE (Centralized Training Decentralized Execution)
        # Takes concatenated observations of all agents during training
        self.centralized_critic = nn.Sequential(
            nn.Linear(obs_dim * n_agents, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode raw observations into hidden representation.

        Args:
            obs: (batch_size, n_agents, obs_dim) or (batch_size, obs_dim)

        Returns:
            hidden: (batch_size, n_agents, hidden_dim) or (batch_size, hidden_dim)
        """
        original_shape = obs.shape
        if len(original_shape) == 3:
            batch_size, n_agents, obs_dim = original_shape
            obs_flat = obs.reshape(batch_size * n_agents, obs_dim)
            hidden = self.obs_encoder(obs_flat)
            hidden = hidden.reshape(batch_size, n_agents, self.hidden_dim)
        else:
            hidden = self.obs_encoder(obs)
        return hidden

    def compute_centralized_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute centralized value estimate for CTDE training.

        Args:
            obs: (batch_size, n_agents, obs_dim) - all agent observations

        Returns:
            centralized_values: (batch_size, n_agents, 1) - value estimates using global info
        """
        batch_size, n_agents, obs_dim = obs.shape
        # Flatten all agent observations into a single vector
        obs_all = obs.reshape(batch_size, n_agents * obs_dim)
        # Compute centralized value (one value per batch, broadcast to all agents)
        central_value = self.centralized_critic(obs_all)  # (batch, 1)
        # Expand to match shape (batch, n_agents, 1) for compatibility
        centralized_values = central_value.unsqueeze(1).expand(batch_size, n_agents, 1)
        return centralized_values

    def generate_thought(self, hidden: torch.Tensor) -> torch.Tensor:
        """Generate thought vector for communication."""
        if not self.use_atoc:
            # return empty thought if ATOC is not used
            return torch.zeros(
                (hidden.shape[0], self.thought_dim), device=hidden.device
            )
        return self.thought_generator(hidden)

    def decide_communication(
        self, hidden: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decide whether to communicate using ATOC gate.

        Args:
            hidden: (batch_size, hidden_dim) or (batch_size, n_agents, hidden_dim)
            deterministic: if True, use gate_prob > 0.5 instead of sampling

        Returns:
            should_communicate: (batch_size,) or (batch_size, n_agents) - binary decision
            gate_probs: (batch_size,) or (batch_size, n_agents) - communication probability
        """
        if not self.use_atoc:
            # Without ATOC, always communicate (or never, depending on your baseline)
            shape = hidden.shape[:-1]
            return torch.ones(shape, device=hidden.device), torch.ones(
                shape, device=hidden.device
            )

        original_shape = hidden.shape
        if len(original_shape) == 3:
            batch_size, n_agents, _ = original_shape
            hidden_flat = hidden.reshape(batch_size * n_agents, self.hidden_dim)
            gate_probs_flat, _, _ = self.attention_unit(
                hidden_flat, return_gate_only=True
            )
            gate_probs = gate_probs_flat.reshape(batch_size, n_agents).squeeze(-1)
        else:
            gate_probs, _, _ = self.attention_unit(hidden, return_gate_only=True)
            gate_probs = gate_probs.squeeze(-1)

        if deterministic:
            should_communicate = (gate_probs > 0.5).float()
        else:
            should_communicate = torch.bernoulli(gate_probs)

        return should_communicate, gate_probs

    def integrate_communication(
        self,
        hidden: torch.Tensor,
        thoughts: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate received thoughts using attention mechanism.

        Args:
            hidden: (batch_size, hidden_dim) - agent's hidden state
            thoughts: (batch_size, n_messages, thought_dim) - received thoughts
            valid_mask: (batch_size, n_messages) - mask for valid messages

        Returns:
            fused_hidden: (batch_size, hidden_dim) - hidden state after communication
            attention_weights: (batch_size, n_messages) - attention over messages
        """
        if not self.use_atoc or thoughts is None:
            return hidden, torch.zeros_like(thoughts[:, :, 0])

        # Apply mask to thoughts if provided (zero out invalid messages)
        if valid_mask is not None:
            thoughts = thoughts * valid_mask.unsqueeze(-1)

        gate_prob, integrated_thought, attention_weights = self.attention_unit(
            hidden, thoughts
        )

        # Fuse own hidden state with integrated communication
        combined = torch.cat([hidden, integrated_thought], dim=-1)
        fused_hidden = self.fusion_layer(combined)

        return fused_hidden, attention_weights

    def forward(
        self,
        obs: torch.Tensor,
        other_thoughts: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Per-agent forward pass. Designed for RLlib with independent agent calls.

        Args:
            obs: (batch_size, obs_dim) - single agent observation
            other_thoughts: (batch_size, n_other_agents, thought_dim) - received thoughts from other agents
            valid_mask: (batch_size, n_other_agents) - mask for valid thoughts
            deterministic: if True, use deterministic actions

        Returns:
            Dictionary containing:
                - action_logits: (batch_size, action_dim)
                - values: (batch_size,)
                - thought: (batch_size, thought_dim) - generated thought for broadcasting
                - comm_decision: (batch_size,) - whether to communicate (binary 0/1)
                - comm_prob: (batch_size,) - probability of communication
                - attention_weights: (batch_size, n_other_agents) - if using ATOC
        """
        # Ensure obs is 2D: (batch, obs_dim)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        batch_size = obs.shape[0]

        # Encode observation
        hidden = self.obs_encoder(obs)  # (batch, hidden_dim)

        # Generate thought for broadcasting to other agents
        agent_thought = torch.zeros(
            (batch_size, self.thought_dim), device=hidden.device
        )
        if self.use_atoc:
            agent_thought = self.generate_thought(hidden)  # (batch, thought_dim)

        # Integrate received thoughts from other agents
        attention_weights = torch.zeros(batch_size, 0)
        if self.use_atoc and other_thoughts is not None and other_thoughts.shape[1] > 0:
            fused_hidden, attention_weights = self.integrate_communication(
                hidden, other_thoughts, valid_mask
            )
            hidden = fused_hidden
        else:
            # No communication or no other agents' thoughts available
            pass

        # Decide whether to communicate
        comm_decision, comm_prob = self.decide_communication(hidden, deterministic)
        # Ensure these are 1D: (batch,)
        if len(comm_decision.shape) > 1:
            comm_decision = comm_decision.squeeze(-1)
        if len(comm_prob.shape) > 1:
            comm_prob = comm_prob.squeeze(-1)

        # Generate action logits
        action_logits = self.action_net(hidden)  # (batch, action_dim)

        # Generate value estimate
        values = self.value_net(hidden)  # (batch, 1)
        values = values.squeeze(-1)  # (batch,)

        return {
            "action_logits": action_logits,
            "values": values,
            "thought": agent_thought,
            "comm_decision": comm_decision,
            "comm_prob": comm_prob,
            "attention_weights": attention_weights,
        }

    def compute_comm_penalty(self, comm_decisions: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty for communication to encourage sparse messaging.

        Args:
            comm_decisions: (batch_size,) - binary communication decisions

        Returns:
            penalty: (batch_size,) - communication penalty
        """
        return self.comm_penalty * comm_decisions

    def select_actions(
        self,
        obs: torch.Tensor,
        thoughts: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample actions from policy.

        Args:
            obs: (batch_size, obs_dim) - single agent observation
            thoughts: (batch_size, n_other_agents, thought_dim) - received thoughts from other
            deterministic: if True, use deterministic actions
        Returns:
            Dictionary containing:
            - actions: (batch_size,) - sampled actions
            - log_probs: (batch_size,) - log probabilities of sampled actions
            - values: (batch_size,) - value estimates
            - comm_decisions: (batch_size,) - binary communication decisions
            - comm_probs: (batch_size,) - probabilities of communication
            - thoughts: (batch_size, thought_dim) - generated thoughts for broadcasting
        """
        output = self.forward(obs, thoughts, deterministic=deterministic)
        action_dist = torch.distributions.Categorical(logits=output["action_logits"])

        if deterministic:
            actions = action_dist.probs.argmax(dim=-1)
        else:
            actions = action_dist.sample()

        log_probs = action_dist.log_prob(actions)

        return {
            "actions": actions,
            "log_probs": log_probs,
            "values": output["values"],
            "comm_decisions": output["comm_decision"],
            "comm_probs": output["comm_prob"],
            "thoughts": output["thought"],
        }


# class RandomAgent(MultiAgentBase):
#     """Returns random actions for the environment.

#     Use `act(observations, readiness_mask)` to select actions.
#     If readiness_mask is None, all agents are allowed to act.
#     """

#     def __init__(self, n_agents: int):
#         self.n_agents = n_agents


class MAPPOAgent(MultiAgentBase):
    """
    MAPPO (Multi-Agent PPO) implementation extending the base class.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        obs_dim: int = 10,
        action_dim: int = 6,
        n_agents: int = 3,
        hidden_dim: int = 128,
        thought_dim: int = 64,
        use_atoc: bool = False,
    ):
        super().__init__(
            config, obs_dim, action_dim, n_agents, hidden_dim, thought_dim, use_atoc
        )

    def get_action_distribution(self, action_logits: torch.Tensor):
        """Get categorical distribution over actions."""
        return torch.distributions.Categorical(logits=action_logits)

    def select_actions(
        self,
        obs: torch.Tensor,
        thoughts: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample actions from policy."""
        output = self.forward(obs, thoughts, deterministic=deterministic)
        action_dist = self.get_action_distribution(output["action_logits"])

        if deterministic:
            actions = action_dist.probs.argmax(dim=-1)
        else:
            actions = action_dist.sample()

        log_probs = action_dist.log_prob(actions)

        return {
            "actions": actions,
            "log_probs": log_probs,
            "values": output["values"],
            "comm_decisions": output["comm_decisions"],
            "comm_probs": output["comm_probs"],
            "thoughts": output["thoughts"],
        }


class MATAgent(MultiAgentBase):
    """
    MAT (Multi-Agent Transformer) implementation extending the base class.

    This adds transformer layers for sequence modeling of multi-agent interactions.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        obs_dim: int = 10,
        action_dim: int = 6,
        n_agents: int = 3,
        hidden_dim: int = 128,
        thought_dim: int = 64,
        use_atoc: bool = False,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__(
            config, obs_dim, action_dim, n_agents, hidden_dim, thought_dim, use_atoc
        )

        # Transformer encoder for processing agent interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        obs: torch.Tensor,
        other_thoughts: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        """Forward pass with transformer processing of agent interactions."""
        batch_size, n_agents, obs_dim = obs.shape

        # Encode observations
        hidden = self.encode_observation(obs)  # (batch, n_agents, hidden_dim)

        # Apply transformer to model agent interactions
        hidden = self.transformer(hidden)  # (batch, n_agents, hidden_dim)

        # Generate thoughts
        agent_thoughts = None
        if self.use_atoc:
            hidden_flat = hidden.reshape(batch_size * n_agents, self.hidden_dim)
            thoughts_flat = self.generate_thought(hidden_flat)
            agent_thoughts = thoughts_flat.reshape(
                batch_size, n_agents, self.thought_dim
            )

        # Integrate communication (same as base class)
        attention_weights = None
        if self.use_atoc and other_thoughts is not None:
            fused_hidden_list = []
            attention_weights_list = []
            for i in range(n_agents):
                agent_hidden = hidden[:, i, :]
                agent_thoughts_received = other_thoughts[:, i, :, :]
                agent_mask = valid_mask[:, i, :] if valid_mask is not None else None

                fused, attn = self.integrate_communication(
                    agent_hidden, agent_thoughts_received, agent_mask
                )
                fused_hidden_list.append(fused)
                if attn is not None:
                    attention_weights_list.append(attn)

            hidden = torch.stack(fused_hidden_list, dim=1)
            if attention_weights_list:
                attention_weights = torch.stack(attention_weights_list, dim=1)

        # Decide communication
        comm_decisions, comm_probs = self.decide_communication(hidden, deterministic)

        # Generate actions and values
        hidden_flat = hidden.reshape(batch_size * n_agents, self.hidden_dim)
        action_logits = self.action_net(hidden_flat)
        action_logits = action_logits.reshape(batch_size, n_agents, self.action_dim)

        # Apply ATOC masking: when ATOC is enabled, mask action 5 (broadcast) based on comm gate
        if self.use_atoc:
            # Add log probability adjustment to action 5 logits
            # When comm_probs is low, adds large negative value to suppress action 5
            logit_adjustment = torch.zeros_like(action_logits)
            logit_adjustment[..., 5] = torch.log(comm_probs + 1e-8)
            action_logits = action_logits + logit_adjustment

        values = self.value_net(hidden_flat)
        values = values.reshape(batch_size, n_agents, 1)

        return {
            "action_logits": action_logits,
            "values": values,
            "comm_decisions": comm_decisions,
            "comm_probs": comm_probs,
            "thoughts": agent_thoughts,
            "attention_weights": attention_weights,
            "hidden": hidden,
        }

    def get_action_distribution(self, action_logits: torch.Tensor):
        """Get categorical distribution over actions."""
        return torch.distributions.Categorical(logits=action_logits)

    def select_actions(
        self,
        obs: torch.Tensor,
        thoughts: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample actions from policy using transformer."""
        output = self.forward(obs, thoughts, deterministic=deterministic)
        action_dist = self.get_action_distribution(output["action_logits"])

        if deterministic:
            actions = action_dist.probs.argmax(dim=-1)
        else:
            actions = action_dist.sample()

        log_probs = action_dist.log_prob(actions)

        return {
            "actions": actions,
            "log_probs": log_probs,
            "values": output["values"],
            "comm_decisions": output["comm_decisions"],
            "comm_probs": output["comm_probs"],
            "thoughts": output["thoughts"],
        }
