#!/usr/bin/env python3
"""Read and summarize training metrics from a PyTorch Lightning experiment directory.

Usage:
    python scripts/read_metrics.py exp/generative/ddpm
    python scripts/read_metrics.py ddpm              # searches ./exp/ for a match
"""

import sys
import os
import json
import glob
from pathlib import Path


def find_exp_dir(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    matches = sorted(Path("exp").rglob(name_or_path)) if Path("exp").exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No experiment directory found for: {name_or_path}")


def read_csv_metrics(exp_dir: Path) -> list[dict]:
    csv_files = list(exp_dir.rglob("metrics.csv"))
    if not csv_files:
        return []
    # Use most recently modified
    csv_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    rows = []
    with open(csv_file) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = line.split(",")
                continue
            values = line.split(",")
            row = {}
            for k, v in zip(header, values):
                try:
                    row[k] = float(v) if v else None
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def read_hparams(exp_dir: Path) -> dict:
    for f in exp_dir.rglob("hparams.yaml"):
        try:
            import yaml
            with open(f) as fp:
                return yaml.safe_load(fp) or {}
        except ImportError:
            with open(f) as fp:
                return {"raw": fp.read()}
    return {}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}

    metric_keys = [k for k in rows[0] if k not in ("step", "epoch")]
    summary = {"total_rows": len(rows)}

    for key in metric_keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        summary[key] = {
            "first": round(vals[0], 6),
            "last": round(vals[-1], 6),
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
            "at_step": None,
        }
        if "val" in key or "loss" in key.lower():
            min_idx = vals.index(min(vals))
            # find the step where this minimum occurred
            non_null = [r for r in rows if r.get(key) is not None]
            if non_null:
                summary[key]["best_step"] = non_null[min_idx].get("step")

    return summary


def format_report(exp_dir: Path, summary: dict, hparams: dict) -> str:
    lines = [
        f"# Metrics Report: {exp_dir}",
        "",
        "## Hyperparameters",
    ]
    if hparams:
        for k, v in list(hparams.items())[:20]:
            lines.append(f"  {k}: {v}")
    else:
        lines.append("  (hparams.yaml not found)")

    lines += ["", "## Metric Summary", ""]
    if not summary:
        lines.append("  No metrics.csv found — training may not have started logging yet.")
    else:
        lines.append(f"  Total logged rows: {summary.get('total_rows', 0)}")
        for key, stats in summary.items():
            if key == "total_rows" or not isinstance(stats, dict):
                continue
            best_info = ""
            if "best_step" in stats and stats["best_step"] is not None:
                best_info = f"  (best at step {int(stats['best_step'])})"
            lines.append(
                f"  {key}: first={stats['first']}  last={stats['last']}  "
                f"min={stats['min']}  max={stats['max']}{best_info}"
            )

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/read_metrics.py <exp_dir_or_name>")
        sys.exit(1)

    try:
        exp_dir = find_exp_dir(sys.argv[1])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    rows = read_csv_metrics(exp_dir)
    hparams = read_hparams(exp_dir)
    summary = summarize(rows)
    print(format_report(exp_dir, summary, hparams))

    # Also emit JSON for programmatic use
    json_path = exp_dir / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "hparams": hparams}, f, indent=2)
    print(f"\n(JSON summary saved to {json_path})")


if __name__ == "__main__":
    main()
