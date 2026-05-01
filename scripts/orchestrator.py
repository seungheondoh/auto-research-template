#!/usr/bin/env python3
"""
Multi-agent research orchestrator.

Main agent (claude-sonnet-4-6) coordinates parallel sub-agents (claude-haiku-4-5)
to analyze each experiment in a sweep, then synthesizes a comparative Markdown report.

Usage:
    python scripts/orchestrator.py                            # all exp/ runs in sweep.config.yaml
    python scripts/orchestrator.py --sweep sweep.config.yaml
    python scripts/orchestrator.py --only flow_matching       # single experiment
    python scripts/orchestrator.py --out research/report.md  # custom output path
"""

import argparse
import concurrent.futures
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("anthropic not installed. Run: uv pip install anthropic")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("pyyaml not installed. Run: uv pip install pyyaml")
    sys.exit(1)

# ── Models ────────────────────────────────────────────────────────────────────

MAIN_MODEL = "claude-sonnet-4-6"
SUB_MODEL = "claude-haiku-4-5-20251001"

# ── Prompts ───────────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM = """You are an ML experiment analyst. You will receive structured data for one
training run and must return a JSON analysis. Be concise and factual — no fluff.

Always return valid JSON matching this schema:
{
  "experiment_id": str,
  "model_type": str,
  "best_val_loss": float | null,
  "final_val_loss": float | null,
  "final_train_loss": float | null,
  "convergence": "converged" | "improving" | "diverged" | "stalled" | "unknown",
  "overfit_risk": "low" | "medium" | "high" | "unknown",
  "hypothesis_verdict": "confirmed" | "partial" | "refuted" | "untestable",
  "key_observations": [str, str, str],
  "top_ablation": str,
  "recommendation": str
}"""

SYNTHESIS_SYSTEM = """You are a senior ML researcher writing a comparative experiment report.
You will receive JSON analyses of multiple experiments from a sweep and must produce
a clear, structured Markdown report that a researcher can act on immediately.

The report must include:
1. Executive summary (3 sentences max)
2. Results table comparing all experiments
3. Per-experiment deep-dive (one paragraph each)
4. Cross-experiment insights (what patterns emerge?)
5. Ranked recommendations for next experiments

Be direct and specific. Cite metric values. Avoid vague statements."""

# ── Data loading ──────────────────────────────────────────────────────────────

def find_exp_dir(model_name: str) -> Path | None:
    base = Path("exp")
    if not base.exists():
        return None
    for d in base.rglob(model_name):
        if d.is_dir():
            return d
    return None


def load_metrics_csv(exp_dir: Path) -> list[dict]:
    csvs = sorted(exp_dir.rglob("metrics.csv"), key=lambda f: f.stat().st_mtime)
    if not csvs:
        return []
    rows, header = [], None
    with open(csvs[-1]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = line.split(",")
                continue
            row = {}
            for k, v in zip(header, line.split(",")):
                try:
                    row[k] = float(v) if v else None
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def summarize_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k not in ("step", "epoch")]
    out = {"n_rows": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            continue
        out[k] = {"first": round(vals[0], 6), "last": round(vals[-1], 6),
                  "min": round(min(vals), 6), "max": round(max(vals), 6)}
    return out


def load_hparams(exp_dir: Path) -> dict:
    for f in exp_dir.rglob("hparams.yaml"):
        with open(f) as fp:
            return yaml.safe_load(fp) or {}
    return {}


def load_monitor_samples(exp_dir: Path, n: int = 3) -> list[dict]:
    monitor_dir = exp_dir / "monitor"
    if not monitor_dir.exists():
        return []
    files = sorted(monitor_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)[-n:]
    samples = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            samples.append({"step": f.stem, "data": data})
        except Exception:
            pass
    return samples


def load_hypothesis(model_name: str) -> str:
    hyp_dir = Path("research/hypotheses")
    if not hyp_dir.exists():
        return ""
    for f in sorted(hyp_dir.glob("*.md")):
        if model_name.lower() in f.stem.lower():
            text = f.read_text()
            # Extract the hypothesis section
            for line in text.splitlines():
                if line.startswith("## Hypothesis"):
                    idx = text.find(line)
                    snippet = text[idx:idx + 500]
                    return snippet.strip()
    return ""


def load_experiment_data(exp_cfg: dict) -> dict:
    model_name = exp_cfg.get("id", exp_cfg.get("model_name", "unknown"))
    exp_dir = find_exp_dir(model_name)

    data = {
        "experiment_id": model_name,
        "model_type": exp_cfg.get("model_type", exp_cfg.get("conditioning_type", "unknown")),
        "description": exp_cfg.get("description", ""),
        "hypothesis": exp_cfg.get("hypothesis", "") or load_hypothesis(model_name),
        "has_data": False,
        "metrics_summary": {},
        "hparams": {},
        "samples": [],
    }

    if exp_dir is None:
        return data

    rows = load_metrics_csv(exp_dir)
    data["metrics_summary"] = summarize_metrics(rows)
    data["hparams"] = load_hparams(exp_dir)
    data["samples"] = load_monitor_samples(exp_dir)
    data["has_data"] = bool(rows)
    return data


# ── Sub-agent: analyze one experiment ─────────────────────────────────────────

def analyze_experiment(client: anthropic.Anthropic, exp_id: str, data: dict) -> dict:
    prompt = (
        f"Analyze this experiment and return JSON only.\n\n"
        f"Experiment data:\n{json.dumps(data, indent=2, default=str)}"
    )
    try:
        response = client.messages.create(
            model=SUB_MODEL,
            max_tokens=1024,
            system=[{"type": "text", "text": ANALYSIS_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rstrip("`").strip()
        return json.loads(text)
    except Exception as e:
        return {
            "experiment_id": exp_id,
            "error": str(e),
            "key_observations": ["Analysis failed"],
            "recommendation": "Re-run after checking experiment data",
        }


# ── Main agent: synthesize report ─────────────────────────────────────────────

def synthesize_report(client: anthropic.Anthropic, analyses: list[dict],
                      sweep_name: str, sweep_description: str) -> str:
    prompt = (
        f"Sweep: {sweep_name}\n"
        f"Description: {sweep_description}\n\n"
        f"Sub-agent analyses ({len(analyses)} experiments):\n"
        f"{json.dumps(analyses, indent=2, default=str)}\n\n"
        f"Write a full comparative Markdown report."
    )
    response = client.messages.create(
        model=MAIN_MODEL,
        max_tokens=4096,
        system=[{"type": "text", "text": SYNTHESIS_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── Report saving ─────────────────────────────────────────────────────────────

def save_report(report: str, out_path: Path, obsidian_path: str = "") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"  Report saved → {out_path}")

    if obsidian_path:
        obs = Path(obsidian_path) / "experiments" / out_path.name
        obs.parent.mkdir(parents=True, exist_ok=True)
        obs.write_text(report)
        print(f"  Obsidian copy → {obs}")
        try:
            import subprocess
            root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], cwd=obsidian_path,
                stderr=subprocess.DEVNULL).decode().strip()
            subprocess.run(["git", "add", str(obs)], cwd=root, check=False)
            subprocess.run(
                ["git", "commit", "-m", f"research: {out_path.stem}"],
                cwd=root, check=False)
            subprocess.run(["git", "push"], cwd=root, check=False)
            print("  Obsidian vault pushed.")
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent experiment orchestrator")
    parser.add_argument("--sweep", default="sweep.config.yaml", help="Sweep config file")
    parser.add_argument("--only", default="", help="Run only this experiment ID")
    parser.add_argument("--out", default="", help="Output report path")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load sweep config
    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        print(f"Sweep config not found: {sweep_path}")
        print("Creating a minimal sweep from exp/ directory...")
        sweep_cfg = {"name": "adhoc", "description": "Ad-hoc sweep", "experiments": []}
        if Path("exp").exists():
            for d in sorted(Path("exp").rglob("*")):
                if d.is_dir() and (d / "metrics.csv").exists():
                    sweep_cfg["experiments"].append({"id": d.name})
    else:
        with open(sweep_path) as f:
            sweep_cfg = yaml.safe_load(f)

    experiments = sweep_cfg.get("experiments", [])
    if args.only:
        experiments = [e for e in experiments if e.get("id") == args.only]
        if not experiments:
            print(f"No experiment with id '{args.only}' in sweep config.")
            sys.exit(1)

    sweep_name = sweep_cfg.get("name", "sweep")
    sweep_description = sweep_cfg.get("description", "")

    print(f"\n{'─'*60}")
    print(f"  Orchestrator: {sweep_name}")
    print(f"  Experiments:  {[e.get('id') for e in experiments]}")
    print(f"  Sub-agent:    {SUB_MODEL}")
    print(f"  Main agent:   {MAIN_MODEL}")
    print(f"{'─'*60}\n")

    # Load all experiment data
    print("Loading experiment data...")
    exp_data = {}
    for exp_cfg in experiments:
        exp_id = exp_cfg.get("id", exp_cfg.get("model_name", "unknown"))
        data = load_experiment_data(exp_cfg)
        exp_data[exp_id] = data
        status = "OK" if data["has_data"] else "no data"
        print(f"  [{status}] {exp_id}")

    # Parallel sub-agent analysis
    print(f"\nSpawning {len(exp_data)} sub-agents in parallel...")
    analyses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(exp_data)) as executor:
        futures = {
            executor.submit(analyze_experiment, client, exp_id, data): exp_id
            for exp_id, data in exp_data.items()
        }
        for future in concurrent.futures.as_completed(futures):
            exp_id = futures[future]
            result = future.result()
            analyses.append(result)
            verdict = result.get("hypothesis_verdict", "?")
            conv = result.get("convergence", "?")
            print(f"  [done] {exp_id} — convergence={conv}, hypothesis={verdict}")

    # Main agent synthesis
    print(f"\nSynthesizing report ({MAIN_MODEL})...")
    report = synthesize_report(client, analyses, sweep_name, sweep_description)

    # Save
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_name = args.out or f"research/experiments/{sweep_name}_report_{date_str}.md"
    out_path = Path(out_name)
    obsidian_path = os.environ.get("OBSIDIAN_VAULT_PATH", "")
    save_report(report, out_path, obsidian_path)

    print(f"\n{'─'*60}")
    print(f"  Done. View: {out_path}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
