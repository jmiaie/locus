#!/usr/bin/env python3
"""
Benchmark regression comparison script.

Compares current benchmark results against baseline.
Fails CI if regression exceeds threshold.

Usage:
    python scripts/benchmark_compare.py current.json baseline.json --regression-threshold 20
"""

import json
import sys
from pathlib import Path
from typing import Any


def load_results(path: str) -> dict[str, Any]:
    """Load benchmark results from JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Baseline {path} not found. Skipping comparison.")
        return {}


def compare_metrics(current: dict, baseline: dict, threshold: float = 20.0) -> tuple[bool, list[str]]:
    """
    Compare current vs baseline metrics.
    
    Returns (passed, messages) where passed=True if within threshold.
    """
    if not baseline:
        return True, ["No baseline to compare against."]
    
    messages = []
    passed = True
    
    # Compare key latency metrics
    metrics_to_compare = [
        ("signal_ablation", "full_6signal", "R@1"),
        ("signal_ablation", "full_6signal", "R@5"),
        ("signal_ablation", "full_6signal", "MRR"),
        ("signal_ablation", "full_6signal", "ms/q"),
    ]
    
    for category, config, metric in metrics_to_compare:
        try:
            current_val = current.get(category, {}).get(config, {}).get(metric)
            baseline_val = baseline.get(category, {}).get(config, {}).get(metric)
            
            if current_val is None or baseline_val is None:
                continue
            
            if metric == "ms/q":
                # For latency, threshold is percentage increase
                pct_change = ((current_val - baseline_val) / baseline_val) * 100
                if pct_change > threshold:
                    messages.append(
                        f"❌ REGRESSION: {category}/{config}/{metric} "
                        f"{baseline_val:.1f}ms → {current_val:.1f}ms (+{pct_change:.1f}%)"
                    )
                    passed = False
                else:
                    messages.append(
                        f"✓ {category}/{config}/{metric}: {current_val:.1f}ms "
                        f"({pct_change:+.1f}%)"
                    )
            else:
                # For Recall/MRR, threshold is percentage decrease
                pct_change = ((current_val - baseline_val) / baseline_val) * 100
                if pct_change < -threshold:
                    messages.append(
                        f"❌ REGRESSION: {category}/{config}/{metric} "
                        f"{baseline_val:.3f} → {current_val:.3f} ({pct_change:.1f}%)"
                    )
                    passed = False
                else:
                    messages.append(
                        f"✓ {category}/{config}/{metric}: {current_val:.3f} "
                        f"({pct_change:+.1f}%)"
                    )
        except (KeyError, TypeError):
            continue
    
    return passed, messages


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare benchmark results against baseline"
    )
    parser.add_argument("current", help="Current benchmark results (JSON)")
    parser.add_argument("baseline", help="Baseline benchmark results (JSON)")
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=20.0,
        help="Threshold for regression (percent)",
    )
    args = parser.parse_args()
    
    current = load_results(args.current)
    baseline = load_results(args.baseline)
    
    passed, messages = compare_metrics(current, baseline, args.regression_threshold)
    
    print("\n=== Benchmark Regression Report ===\n")
    for msg in messages:
        print(msg)
    print()
    
    if passed:
        print("✓ All benchmarks within threshold.")
        return 0
    else:
        print("✗ Benchmark regression detected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
