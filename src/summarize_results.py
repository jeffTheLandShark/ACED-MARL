import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_records(result_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not result_path.exists():
        return records
    with result_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _metric_from_record(rec: Dict[str, Any]) -> Optional[float]:
    # Prefer new-style key first
    if "env_runners/episode_return_mean" in rec:
        return rec.get("env_runners/episode_return_mean")
    # Fallback to nested env_runners payload
    env_runners = rec.get("env_runners", {}) or {}
    for key in ("episode_return_mean", "episode_reward_mean"):
        if key in env_runners:
            return env_runners.get(key)
    return None


def _episode_len_from_record(rec: Dict[str, Any]) -> Optional[float]:
    if "env_runners/episode_len_mean" in rec:
        return rec.get("env_runners/episode_len_mean")
    env_runners = rec.get("env_runners", {}) or {}
    if "episode_len_mean" in env_runners:
        return env_runners.get("episode_len_mean")
    return None


def _best_and_last(
    records: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not records:
        return None, None
    best = None
    best_val = float("-inf")
    for r in records:
        val = _metric_from_record(r)
        if val is None:
            continue
        if val > best_val:
            best_val = val
            best = r
    last = records[-1]
    return best, last


def _find_checkpoints(trial_dir: Path) -> List[Path]:
    return sorted(trial_dir.glob("checkpoint_*"))


def _extract_custom_metrics(rec: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not rec:
        return {}
    cm = rec.get("custom_metrics", {}) or {}

    # Some Ray payloads may prefix keys with "custom_metrics/"; normalize them.
    normalized = {}
    for k, v in cm.items():
        key = k
        if k.startswith("custom_metrics/"):
            key = k.split("/", 1)[1]
        # Keep only numeric values
        if isinstance(v, (int, float)):
            normalized[key] = float(v)
    return normalized


def _extract_custom_metrics_from_progress(trial_dir: Path) -> Dict[str, float]:
    progress_path = trial_dir / "progress.csv"
    if not progress_path.exists():
        return {}

    last_row: Optional[Dict[str, str]] = None
    with progress_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row

    if not last_row:
        return {}

    metrics: Dict[str, float] = {}
    for k, v in last_row.items():
        if "custom_metrics" not in k:
            continue
        try:
            metrics[k.split("/")[-1]] = float(v)
        except (TypeError, ValueError):
            continue
    return metrics


def summarize_trial(trial_dir: Path) -> Dict[str, Any]:
    result_path = trial_dir / "result.json"
    records = _load_records(result_path)
    best, last = _best_and_last(records)

    latest_ckpts = _find_checkpoints(trial_dir)
    latest_ckpt = latest_ckpts[-1] if latest_ckpts else None

    def safe_get(rec: Optional[Dict[str, Any]], key: str, default=None):
        if rec is None:
            return default
        return rec.get(key, default)

    json_metrics = _extract_custom_metrics(last)
    csv_metrics = _extract_custom_metrics_from_progress(trial_dir)
    merged_metrics = {**json_metrics, **csv_metrics}

    return {
        "trial": trial_dir.name,
        "iterations": len(records),
        "best_return_mean": _metric_from_record(best) if best else None,
        "best_iteration": safe_get(best, "training_iteration"),
        "last_return_mean": _metric_from_record(last) if last else None,
        "last_episode_len_mean": _episode_len_from_record(last) if last else None,
        "total_env_steps": safe_get(last, "num_env_steps_sampled_lifetime"),
        "total_agent_steps": safe_get(last, "num_agent_steps_sampled_lifetime"),
        "last_timestamp": safe_get(last, "date"),
        "checkpoint": str(latest_ckpt) if latest_ckpt else None,
        "custom_metrics": merged_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize Ray Tune PPO results.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("results/quick_test"),
        help="Path to experiment root (contains trial folders).",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists():
        # Fallback: common alternative layout under src/results
        alt_root = Path("src") / root
        if alt_root.exists():
            root = alt_root
        else:
            print(f"No such directory: {root}")
            return

    trial_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("PPO_")]
    if not trial_dirs:
        # Try one-level deeper (e.g., root/quick_test/PPO_*)
        nested = []
        for sub in root.iterdir():
            if sub.is_dir():
                nested.extend(
                    [
                        p
                        for p in sub.iterdir()
                        if p.is_dir() and p.name.startswith("PPO_")
                    ]
                )
        trial_dirs = nested
    if not trial_dirs:
        print(f"No trial directories found under {root}")
        return

    summaries = [summarize_trial(td) for td in sorted(trial_dirs)]

    print("Summary (per trial):")
    for s in summaries:
        print("-" * 80)
        print(f"trial:            {s['trial']}")
        print(f"iterations:       {s['iterations']}")
        print(f"best_return_mean: {s['best_return_mean']}")
        print(f"best_iteration:   {s['best_iteration']}")
        print(f"last_return_mean: {s['last_return_mean']}")
        print(f"last_ep_len_mean: {s['last_episode_len_mean']}")
        print(f"env_steps_total:  {s['total_env_steps']}")
        print(f"agent_steps_total:{s['total_agent_steps']}")
        print(f"last_timestamp:   {s['last_timestamp']}")
        print(f"latest_checkpoint:{s['checkpoint']}")
        if s["custom_metrics"]:
            print("custom_metrics:")
            for k, v in sorted(s["custom_metrics"].items()):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
