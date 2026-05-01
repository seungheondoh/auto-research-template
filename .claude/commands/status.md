# /status — Check Training Status

Show the current state of a training run (or all recent runs).

**Input:** $ARGUMENTS
Examples:
- `/status` → most recent run across all exp/
- `/status ddpm` → specific experiment by model_name
- `/status generative/flow_matching` → by project/model_name path

## Step-by-Step Instructions

### 1. Find the experiment

**If a name is given:**
```bash
find ./exp -type d -name "{name}" 2>/dev/null | head -5
```

**If no name:**
```bash
ls -lt ./exp/*/ 2>/dev/null | head -10   # most recently modified
```

Pick the most recently active experiment directory.

### 2. Gather status signals

Run all of these:
```bash
# Last checkpoint mtime
ls -lh ./exp/{project}/{model_name}/last.ckpt 2>/dev/null

# Step count from latest monitor output
ls -t ./exp/{project}/{model_name}/monitor/*.json 2>/dev/null | head -1

# Latest metrics from CSV
tail -5 ./exp/{project}/{model_name}/metrics.csv 2>/dev/null

# Is training still running?
ps aux | grep "train.py" | grep -v grep
```

### 3. Format status report

Print exactly this format:

```
Experiment     : {model_name}
Project        : {project}
Trainer type   : {trainer_type}
─────────────────────────────────────────
Last step      : {step from metrics.csv}
Latest val_loss: {val_loss}
Latest train_loss: {train_loss}
─────────────────────────────────────────
Checkpoint     : {mtime of last.ckpt or "not found"}
Training PID   : {PID if running, else "not running"}
─────────────────────────────────────────
Next:  /results {model_name}   — full results
       /analyze {model_name}   — multi-agent analysis
```

### 4. Latest generated sample (if any)

If `monitor/*.json` exists, show the most recent entry:
```
Latest sample (step {step}):
  "{generated text or description}"
```
