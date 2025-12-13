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

    def __init__(self, hidden_dim: int, broadcast_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.broadcast_dim = broadcast_dim

        # Gate network: decides probability of initiating communication
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Query and key networks for attention mechanism
        self.query_net = nn.Linear(hidden_dim, broadcast_dim)
        self.key_net = nn.Linear(broadcast_dim, broadcast_dim)
        self.value_net = nn.Linear(broadcast_dim, broadcast_dim)

    def forward(
        self,
        hidden_state: torch.Tensor,
        broadcasts: Optional[torch.Tensor] = None,
        return_gate_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_state: (batch_size, hidden_dim) - agent's internal representation
            broadcasts: (batch_size, n_other_agents, broadcast_dim) - received broadcast vectors
            return_gate_only: if True, only compute communication gate

        Returns:
            gate_prob: (batch_size, 1) - probability of initiating communication
            integrated_broadcast: (batch_size, broadcast_dim) - attention-weighted message integration
            attention_weights: (batch_size, n_other_agents) - attention over received messages
        """
        # Communication gate: should this agent broadcast?
        gate_prob = self.gate_net(hidden_state)

        if return_gate_only or broadcasts is None:
            return gate_prob, None, None

        # Attention over received broadcasts
        batch_size = hidden_state.shape[0]
        n_messages = broadcasts.shape[1] if len(broadcasts.shape) == 3 else 1

        # Query from own hidden state
        query = self.query_net(hidden_state)  # (batch, broadcast_dim)

        # Keys and values from received broadcasts
        if len(broadcasts.shape) == 2:
            broadcasts = broadcasts.unsqueeze(1)  # (batch, 1, broadcast_dim)

        keys = self.key_net(broadcasts)  # (batch, n_messages, broadcast_dim)
        values = self.value_net(broadcasts)  # (batch, n_messages, broadcast_dim)

        # Scaled dot-product attention
        scores = torch.bmm(
            query.unsqueeze(1),  # (batch, 1, broadcast_dim)
            keys.transpose(1, 2),  # (batch, broadcast_dim, n_messages)
        ) / np.sqrt(
            self.broadcast_dim
        )  # (batch, 1, n_messages)

        attention_weights = F.softmax(scores, dim=-1)  # (batch, 1, n_messages)

        # Weighted sum of values
        integrated_broadcast = torch.bmm(
            attention_weights,  # (batch, 1, n_messages)
            values,  # (batch, n_messages, broadcast_dim)
        ).squeeze(
            1
        )  # (batch, broadcast_dim)

        return gate_prob, integrated_broadcast, attention_weights.squeeze(1)


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
        hidden_dim: int = 128,
        broadcast_dim: int = 64,
        use_atoc: bool = False,
        comm_penalty: float = 0.01,
    ):
        super().__init__()
        self.obs_dim = obs_dim  # Base observation: 6
        self.broadcast_dim = broadcast_dim  # 64
        self.max_other_agents = max_other_agents  # n_agents - 1
        self.extended_obs_dim = obs_dim + (max_other_agents * broadcast_dim)
        self.hidden_dim = hidden_dim
        self.broadcast_dim = broadcast_dim
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
            self.attention_unit = AttentionUnit(hidden_dim, broadcast_dim)
            # Broadcast generator: creates message content from hidden state
            self.broadcast_generator = nn.Sequential(
                nn.Linear(hidden_dim, broadcast_dim), nn.Tanh()
            )
            # Fusion layer: combines own hidden state with integrated messages
            self.fusion_layer = nn.Linear(hidden_dim + broadcast_dim, hidden_dim)

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

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode raw observations into hidden representation.

        Args:
            obs: (batch_size, obs_dim)

        Returns:
            hidden: (batch_size, hidden_dim)
        """
        return self.obs_encoder(obs)

    def generate_broadcast(self, hidden: torch.Tensor) -> torch.Tensor:
        """Generate broadcast message vector for communication (action 5 only)."""
        if not self.use_atoc:
            # return empty broadcast if ATOC is not used
            return torch.zeros(
                (hidden.shape[0], self.broadcast_dim), device=hidden.device
            )
        return self.broadcast_generator(hidden)

    def decide_communication(
        self, hidden: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decide whether to communicate using ATOC gate.

        Args:
            hidden: (batch_size, hidden_dim)
            deterministic: if True, use gate_prob > 0.5 instead of sampling

        Returns:
            should_communicate: (batch_size,) - binary decision
            gate_probs: (batch_size,) - communication probability
        """
        if not self.use_atoc:
            # Without ATOC, always communicate (or never, depending on your baseline)
            shape = hidden.shape[:-1]
            return torch.ones(shape, device=hidden.device), torch.ones(
                shape, device=hidden.device
            )

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
        broadcasts: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate received broadcasts using attention mechanism.

        Args:
            hidden: (batch_size, hidden_dim) - agent's hidden state
            broadcasts: (batch_size, n_messages, broadcast_dim) - received broadcasts
            valid_mask: (batch_size, n_messages) - mask for valid messages

        Returns:
            fused_hidden: (batch_size, hidden_dim) - hidden state after communication
            attention_weights: (batch_size, n_messages) - attention over messages
        """
        if not self.use_atoc or broadcasts is None:
            return hidden, torch.zeros_like(broadcasts[:, :, 0])

        # Apply mask to broadcasts if provided (zero out invalid messages)
        if valid_mask is not None:
            broadcasts = broadcasts * valid_mask.unsqueeze(-1)

        gate_prob, integrated_broadcast, attention_weights = self.attention_unit(
            hidden, broadcasts
        )

        # Fuse own hidden state with integrated communication
        combined = torch.cat([hidden, integrated_broadcast], dim=-1)
        fused_hidden = self.fusion_layer(combined)

        return fused_hidden, attention_weights

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
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

        # Generate broadcast message
        agent_broadcast = torch.zeros(
            (batch_size, self.broadcast_dim), device=hidden.device
        )
        if self.use_atoc:
            agent_broadcast = self.generate_broadcast(hidden)

        # Integrate received broadcasts
        attention_weights = torch.zeros(batch_size, 0)
        if self.use_atoc and valid_mask.sum() > 0:
            fused_hidden, attention_weights = self.integrate_communication(
                hidden, broadcasts, valid_mask
            )
            hidden = fused_hidden

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
            "broadcast": agent_broadcast,
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

    def get_action_distribution(self, action_logits: torch.Tensor):
        """Get categorical distribution over actions."""
        return torch.distributions.Categorical(logits=action_logits)

    def select_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Sample actions with broadcasts embedded in obs.

        Args:
            obs: (batch_size, extended_obs_dim)
            deterministic: if True, use deterministic actions
        Returns:
        Dictionary containing:
            - actions: (batch_size,) - sampled actions
            - log_probs: (batch_size,) - log probabilities of sampled actions
            - values: (batch_size,) - value estimates
            - comm_decisions: (batch_size,) - binary communication decisions
            - comm_probs: (batch_size,) - probabilities of communication
            - broadcast: (batch_size, broadcast_dim) - broadcast message
        """
        output = self.forward(obs, deterministic=deterministic)
        action_logits = output["action_logits"].clone()

        # ATOC gating: boost action 5 by comm_prob
        if self.use_atoc:
            action_logits[..., 5] = action_logits[..., 5] + torch.log(
                output["comm_prob"] + 1e-8
            )

        action_dist = self.get_action_distribution(action_logits)

        if deterministic:
            action = action_dist.probs.argmax(dim=-1)
        else:
            action = action_dist.sample()

        log_probs = action_dist.log_prob(action)

        return {
            "action": action,
            "log_prob": log_probs,
            "value": output["values"],
            "comm_decision": output["comm_decision"],
            "comm_prob": output["comm_prob"],
            "broadcast": output["broadcast"],
        }


class MAPPOAgent(MultiAgentBase):
    """
    MAPPO (Multi-Agent PPO) implementation extending the base class.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        obs_dim: int = 10,
        action_dim: int = 6,
        hidden_dim: int = 128,
        broadcast_dim: int = 64,
        use_atoc: bool = False,
    ):
        super().__init__(
            config, obs_dim, action_dim, hidden_dim, broadcast_dim, use_atoc
        )


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
        broadcast_dim: int = 64,
        use_atoc: bool = False,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__(
            config, obs_dim, action_dim, n_agents, hidden_dim, broadcast_dim, use_atoc
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
        other_broadcasts: Optional[torch.Tensor] = None,
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
        if self.use_atoc and other_broadcasts is not None:
            fused_hidden_list = []
            attention_weights_list = []
            for i in range(n_agents):
                agent_hidden = hidden[:, i, :]
                agent_thoughts_received = other_broadcasts[:, i, :, :]
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
        broadcasts: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample actions from policy using transformer."""
        output = self.forward(obs, broadcasts, deterministic=deterministic)
        action_dist = self.get_action_distribution(output["action_logits"])

        if deterministic:
            action = action_dist.probs.argmax(dim=-1)
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": output["values"],
            "comm_decision": output["comm_decision"],
            "comm_prob": output["comm_prob"],
            "broadcast": output["broadcast"],
        }
