# /results — Show Experiment Results

Display a formatted comparison of all completed (or in-progress) experiments.

**Input:** $ARGUMENTS
Examples:
- `/results` → all experiments in sweep.config.yaml
- `/results ddpm flow_matching` → specific experiments only
- `/results --report` → also run the multi-agent orchestrator report

## Step-by-Step Instructions

### 1. Discover experiments

**From sweep.config.yaml (preferred):**
```bash
cat sweep.config.yaml
```
Extract all `experiments[].id` values.

**Fallback — scan exp/ directly:**
```bash
find ./exp -name "metrics.csv" | sort
```

Filter to whichever experiments were requested in `$ARGUMENTS` (if any).

### 2. Load metrics for each experiment

For each experiment found, run:
```bash
python scripts/read_metrics.py {exp_dir}
```

Collect: `model_name`, `trainer_type/model_type`, `best_val_loss` (step), `final_val_loss`, `final_train_loss`, total steps.

### 3. Print comparison table

```
## Experiment Results — {date}

| Experiment     | Type           | Best Val Loss | at Step | Final Val | Final Train | Status  |
|----------------|----------------|--------------|---------|-----------|-------------|---------|
| ddpm           | generative     | 0.2341       | 8400    | 0.2389    | 0.1923      | done    |
| flow_matching  | generative     | 0.2108       | 9200    | 0.2201    | 0.1701      | done    |
| drift          | generative     | 0.2290       | 7800    | 0.2310    | 0.1850      | running |

Winner (lowest best_val_loss): {winner}
```

### 4. Hypothesis verdict (if available)

Check `research/experiments/*/analysis.md` for hypothesis verdicts. If found, add:

```
## Hypothesis Verdicts

| Experiment    | Verdict    | Key Finding |
|---------------|-----------|-------------|
| ddpm          | confirmed  | ... |
| flow_matching | partial    | ... |
```

### 5. Multi-agent report (if --report flag)

If `--report` is in `$ARGUMENTS` and `ANTHROPIC_API_KEY` is set:
```bash
python scripts/orchestrator.py
```

Then show the output path of the generated report.

Otherwise, suggest:
```
For a full multi-agent comparative analysis:
  python scripts/orchestrator.py
  /analyze {model_name}
```
