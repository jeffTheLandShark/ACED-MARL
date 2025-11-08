# ACED-MARL — Asynchronous Cooperative Event-Driven Multi-Agent Reinforcement Learning

> Milwaukee School of Engineering (MSOE)<br>
> Leigh Goetsch<br>
> CSC5661 - Reinforcement Learning<br>
> Fall 2025

## Project Overview

This project investigates how asynchronous, event-driven action selection affects coordination in multi-agent reinforcement learning (MARL) environments with communication constraints. The project models a fleet of small rovers collaboratively transporting a shared payload toward a goal location. Each rover operates with partial observability and limited communication bandwidth, balancing between acting on local state information and broadcasting updates to maintain group awareness.

The core objective is to analyze the impact of asynchronous and event-triggered decision-making under latency, packet loss, and sparse communication conditions.

## Research Questions

- How does asynchronous, event-driven action selection affect coordination and task success?
- How will the system respond to factors such as low sampling rates, communication dropouts, and latency? 
- Will a model perform better under a step-based or event-driven action scheme when such obstacles are introduced?

## Algorithms and Variants

- **MAPPO (Multi-Agent Proximal Policy Optimization)** — baseline PPO-based control.
- **MAT (Multi-Agent Transformer)** — sequence-based policy modeling.
- **ATOC (Attentional Communication)** — adaptive learned communication.

| Architecture              | Action Scheme |
|---------------------------|---------------|
| Synchronous MAPPO         | Step-based    |
| Synchronous MAPPO + ATOC  | Step-based    |
| Synchronous MAT           | Step-based    |
| Synchronous MAT + ATOC    | Step-based    |
| Asynchronous MAPPO        | Event-driven  |
| Asynchronous MAT          | Event-driven  |
| Asynchronous MAPPO + ATOC | Event-driven  |
| Asynchronous MAT + ATOC   | Event-driven  |



## References

- Wen, M., Kuba, J. G., Lin, R., Zhang, W., Wen, Y., Wang, J., & Yang, Y. (2022). *Multi-agent reinforcement learning is a sequence modeling problem.* [arXiv:2205.14953](https://arxiv.org/abs/2205.14953)
- Jiang, J., & Lu, Z. (2018). *Learning attentional communication for multi-agent cooperation.* [arXiv:1805.07733](https://arxiv.org/abs/1805.07733)
- Ma, Y., Liu, Y., Zhao, L., & Zhao, M. (2022). *A Review on Cooperative Control Problems of Multi-agent Systems.* IEEE Chinese Control Conference (CCC), 4831–4836. [DOI:10.23919/CCC55666.2022.9902761](https://ieeexplore.ieee.org/document/9902761)
- *Asynchronous Cooperative Multi-Agent Reinforcement Learning with Limited Communication.* [arXiv:2502.00558](https://arxiv.org/abs/2502.00558)
