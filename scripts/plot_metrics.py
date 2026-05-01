#!/usr/bin/env python3
"""Generate standard diagnostic plots from a PyTorch Lightning experiment directory.

Usage:
    python scripts/plot_metrics.py --exp_dir exp/generative/ddpm --out_dir research/experiments/ddpm/figures
"""

import argparse
import sys
from pathlib import Path


def load_csv(exp_dir: Path) -> tuple[list[str], list[dict]]:
    csv_files = list(exp_dir.rglob("metrics.csv"))
    if not csv_files:
        return [], []
    csv_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    rows, header = [], None
    with open(csv_file) as f:
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
    return header or [], rows


def extract_series(rows: list[dict], x_key: str, y_key: str) -> tuple[list, list]:
    pairs = [(r[x_key], r[y_key]) for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    xs, ys = zip(*pairs) if pairs else ([], [])
    return list(xs), list(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not exp_dir.exists():
        # Try searching under ./exp/
        matches = sorted(Path("exp").rglob(args.exp_dir)) if Path("exp").exists() else []
        if matches:
            exp_dir = matches[0]
        else:
            print(f"Error: experiment directory not found: {args.exp_dir}")
            sys.exit(1)

    header, rows = load_csv(exp_dir)
    if not rows:
        print(f"No metrics.csv found under {exp_dir}. Cannot generate plots.")
        sys.exit(1)

    generated = []

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label, color in [("train_loss", "Train Loss", "steelblue"), ("val_loss", "Val Loss", "tomato")]:
        xs, ys = extract_series(rows, "step", key)
        if xs:
            ax.plot(xs, ys, label=label, color=color)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curve — {exp_dir.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / "loss_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path))

    # --- LR schedule ---
    lr_keys = [k for k in header if "lr" in k.lower() or "learning_rate" in k.lower()]
    if lr_keys:
        fig, ax = plt.subplots(figsize=(8, 3))
        for lk in lr_keys:
            xs, ys = extract_series(rows, "step", lk)
            if xs:
                ax.plot(xs, ys, label=lk)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"LR Schedule — {exp_dir.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = out_dir / "lr_schedule.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(path))

    # --- All val metrics ---
    val_keys = [k for k in header if k.startswith("val_") and k != "val_loss"]
    if val_keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        for vk in val_keys:
            xs, ys = extract_series(rows, "step", vk)
            if xs:
                ax.plot(xs, ys, label=vk)
        ax.set_xlabel("Step")
        ax.set_title(f"Validation Metrics — {exp_dir.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = out_dir / "val_metrics.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(path))

    # --- 2x2 summary dashboard ---
    plot_specs = [
        ("train_loss", "Train Loss", "steelblue"),
        ("val_loss", "Val Loss", "tomato"),
    ]
    if lr_keys:
        plot_specs.append((lr_keys[0], "LR", "seagreen"))
    if val_keys:
        plot_specs.append((val_keys[0], val_keys[0], "darkorange"))

    n = len(plot_specs)
    cols = 2
    rows_grid = (n + 1) // cols
    fig, axes = plt.subplots(rows_grid, cols, figsize=(12, 4 * rows_grid))
    axes = [axes] if rows_grid == 1 and cols == 1 else (axes.flat if hasattr(axes, "flat") else [axes])
    for ax, (key, label, color) in zip(axes, plot_specs):
        xs, ys = extract_series(rows, "step", key)
        if xs:
            ax.plot(xs, ys, color=color)
        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
    for ax in list(axes)[len(plot_specs):]:
        ax.set_visible(False)
    fig.suptitle(f"Summary — {exp_dir.name}", fontsize=13)
    fig.tight_layout()
    path = out_dir / "summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path))

    print("Generated figures:")
    for g in generated:
        print(f"  {g}")


if __name__ == "__main__":
    main()
