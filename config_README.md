# ACED-MARL Training Configuration & Job Management

This directory contains configuration files and job submission scripts for training ACED-MARL agents on cluster systems.

## Quick Start

### List all presets
```bash
python train.py --list-presets
```

### Get help
```bash
python train.py --help
```

### Train with preset
```bash
python train.py --preset mappo_sync
```

### Train with config file
```bash
python train.py --config configs/mat_atoc_event.yaml
```

### Override parameters
```bash
python train.py --preset quick_test --override training.num_workers=4
```

### Multiple overrides
```bash
python train.py --preset mappo_sync \
  --override training.num_workers=8 \
  --override training.batch_size=1024 \
  --override training.learning_rate=5e-4
```

## Configuration Files

All configs are in `configs/` directory as YAML files:

| File                    | Description                  | Training Time* |
|-------------------------|------------------------------|----------------|
| `quick_test.yaml`       | Minimal config for debugging | ~5 min         |
| `mappo_sync.yaml`       | MAPPO baseline, synchronous  | ~4 hours       |
| `mappo_event.yaml`      | MAPPO baseline, event-driven | ~4 hours       |
| `mappo_atoc_sync.yaml`  | MAPPO + ATOC, synchronous    | ~4.5 hours     |
| `mappo_atoc_event.yaml` | MAPPO + ATOC, event-driven   | ~4.5 hours     |
| `mat_sync.yaml`         | MAT baseline, synchronous    | ~5 hours       |
| `mat_event.yaml`        | MAT baseline, event-driven   | ~5 hours       |
| `mat_atoc_sync.yaml`    | MAT + ATOC, synchronous      | ~5.5 hours     |
| `mat_atoc_event.yaml`   | MAT + ATOC, event-driven     | ~5.5 hours     |

*Estimates on MSOE GPU cluster (1× GPU, 4 workers)

## Configuration Structure


```yaml
name: experiment_name              # Unique identifier
description: "..."                # What this variant tests

environment:                       # AcedPayloadEnv parameters
  n_agents: 5                     # Number of agents
  arena_size: 25.0                # Arena dimension (meters)
  max_steps: 2000                 # Episode length
  event_driven: true              # Use event-driven cooldown
  cooldown_time: 3                # Steps between broadcasts
  contact_radius: 3.0             # Distance to trigger contact
  comm_dropout_prob: 0.0          # Message loss probability
  comm_delay_steps: 0             # Message delay (steps)
  sensor_dropout_prob: 0.0        # Sensor failure probability

agent:                            # Agent architecture
  agent_type: mappo|mat           # Policy network type
  use_atoc: true|false            # Enable communication
  hidden_dim: 128                 # Network hidden layer size
  thought_dim: 64                 # Communication embedding dim
  n_heads: 4                      # (MAT only) Attention heads
  n_layers: 2                     # (MAT only) Transformer layers

training:                         # PPO training parameters
  num_workers: 4                  # Ray worker processes
  num_envs_per_worker: 4          # Parallel envs per worker
  batch_size: 512                 # Training batch size
  learning_rate: 3e-4             # Optimizer learning rate
  gamma: 0.99                     # Discount factor
  lambda: 0.95                    # GAE lambda
  entropy_coeff: 0.01             # Entropy regularization
  num_epochs: 20                  # PPO update epochs
  total_steps: 1_000_000          # Total environment steps
  checkpoint_freq: 10000          # Save checkpoint every N steps

evaluation:                       # Periodic evaluation
  eval_episodes: 10               # Episodes per eval
  eval_frequency: 50000           # Evaluate every N steps

output_dir: results               # Root output directory
seed: 42                          # Random seed
verbose: true                     # Detailed logging
```


## Output Directory Structure

After training, results are organized as:

```
results/
├── mappo_sync/
│   ├── config.yaml                    # Saved config
│   ├── training_metrics.json          # Loss, reward curves
│   ├── checkpoints/
│   │   ├── best_model.pt              # Best checkpoint
│   │   ├── latest_model.pt            # Latest checkpoint
│   │   └── checkpoint_50000.pt
│   └── evaluation/
│       ├── success_rate.json
│       └── metrics.csv
├── mappo_event/
│   └── ...
└── quick_test/
    └── ...
```

## Experimental Workflow

### Step 1: Test Configuration (5 min)
```bash
python train.py --preset quick_test
```
Verify code runs without errors.

### Step 2: Debug Specific Variant (30 min)
```bash
python train.py --preset mappo_sync --override training.total_steps=100000
```
Ensure specific variant works on your system.

### Step 3: Submit Full Batch (4-6 hours)
```bash
sbatch submit_job.sh mappo_sync
sbatch submit_job.sh mappo_event
sbatch submit_job.sh mat_sync
sbatch submit_job.sh mat_event
```

Or submit all at once:
```bash
bash submit_batch.sh baselines
```

### Step 4: Monitor Jobs
```bash
watch squeue -u $(whoami) -p gpu
tail -f slurm_logs/LATEST_JOB_ID.out
```

### Step 5: Evaluate Results
```bash
python eval.py --checkpoint results/mappo_sync/best_model.pt
```


### Out of Memory
```bash
# Reduce workers and parallel envs
sbatch submit_job.sh mappo_sync \
  --override training.num_workers=2 \
  --override training.num_envs_per_worker=2

# Or reduce batch size
sbatch submit_job.sh mappo_sync \
  --override training.batch_size=256
```

