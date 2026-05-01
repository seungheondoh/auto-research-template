# /analyze — Analyze Experiment Results

Analyze the results of a completed (or in-progress) training run using a three-reviewer sub-agent panel.

**Input:** $ARGUMENTS  
(Experiment name like `ddpm`, or a path like `exp/generative/ddpm`)

## Step-by-Step Instructions

### 1. Locate experiment artifacts

Resolve the experiment directory:
- If a bare name is given (e.g. `ddpm`), search `./exp/` recursively for matching directories
- Expected structure: `exp/{project}/{model_name}/`

Run this to gather artifacts:
```bash
python scripts/read_metrics.py {exp_dir}
```

Also check for:
- `{exp_dir}/metrics.csv` — PyTorch Lightning CSVLogger output
- `{exp_dir}/hparams.yaml` — hyperparameters
- Any `*.ckpt` files for checkpoint info

If no metrics exist yet, report "Training may still be in progress" and show what's available.

### 2. Load the linked hypothesis (if any)

Search `research/hypotheses/` for a `.md` file whose slug matches the experiment name. If found, extract the original hypothesis and the predicted outcome.

### 3. Spawn three analysis agents IN PARALLEL

Pass the full metrics summary and hypothesis (if available) to each agent.

---

**Agent A — Statistical Reviewer**

Prompt:
> You are analyzing ML training metrics. Here are the metrics: {metrics_summary}.
> Assess: (1) Did training converge? Describe the loss curve shape. (2) Is there a train/val gap indicating overfitting or underfitting? (3) At what step did the best val_loss occur? (4) Are the results statistically meaningful or noisy? Return: convergence verdict, overfitting diagnosis, best checkpoint step, confidence in results (low/medium/high).

---

**Agent B — Ablation Reviewer**

Prompt:
> You are planning the next round of ablations for this experiment. Metrics: {metrics_summary}. Config: {hparams_summary}.
> Assess: (1) Which hyperparameters most likely drove the result (good or bad)? (2) What 3 ablations would most efficiently diagnose remaining uncertainty? (3) What would a +/- 20% performance change look like, and what config changes could achieve it? Return: top driver, 3 ablation proposals with expected direction of change, a ranked priority list.

---

**Agent C — Hypothesis Comparison Reviewer**

Prompt:
> Compare the experiment outcome to the original hypothesis prediction. Hypothesis: "{hypothesis_text}" (or "no linked hypothesis" if none). Metrics: {metrics_summary}.
> Assess: (1) Was the hypothesis confirmed, partially confirmed, or refuted? (2) What was surprising vs expected? (3) What does this tell us about the underlying mechanism? Return: verdict (confirmed/partial/refuted), key surprises, mechanistic insight.

---

### 4. Synthesize

Combine the three reviews into:
- **Result summary**: 2-3 sentences on what happened
- **Key finding**: the single most important takeaway
- **Recommended next step**: specific next experiment or ablation

### 5. Save analysis report

Write `research/experiments/{model_name}/analysis.md`:

```markdown
# Analysis: {model_name}

**Date:** {YYYY-MM-DD}
**Exp dir:** {exp_dir}
**Linked hypothesis:** {hypothesis_file or "none"}

## Metrics Summary
{key metrics table}

## Statistical Review
{Agent A output}

## Ablation Proposals
{Agent B output}

## Hypothesis Comparison
{Agent C output}

## Synthesis
{your synthesis}

## Recommended Next Step
{specific next experiment}

## Next Command
/iterate
```

Create `research/experiments/{model_name}/` if needed.

### 6. Update research log

Append to `research/log.md`:
```markdown
### Analysis complete — {YYYY-MM-DD}
- Result: {one-line summary}
- Key finding: {key finding}
- Next step: {next step}
```

### 7. Present to user

Show the synthesis and recommended next step. Suggest `/visualize {model_name}` to generate plots, or `/iterate` to plan the next cycle.
