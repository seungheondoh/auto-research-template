# /train — Start a Training Run

Launch a training run from a config file, with pre-flight checks and monitoring instructions.

**Input:** $ARGUMENTS
Examples:
- `/train` → uses sweep.config.yaml first experiment
- `/train config/generative/flow_matching.yaml`
- `/train config/generative/ddpm.yaml --gpu 0`
- `/train config/lm/default.yaml --resume ./exp/lm/default/last.ckpt`

## Step-by-Step Instructions

### 1. Resolve config

Parse `$ARGUMENTS` to extract:
- `config` path (required; if omitted, take the first experiment from `sweep.config.yaml`)
- `--gpu` value (optional)
- `--resume` path (optional)

Read the config file with:
```bash
cat {config_path}
```

Extract key fields: `project`, `model_name`, `steps`, trainer_type, model_type, batch_size, lr.

### 2. Pre-flight checks

Run these checks:
```bash
# Data directory
ls {data_dir}

# Check for existing run (resume candidate)
ls -lh ./exp/{project}/{model_name}/ 2>/dev/null || echo "no prior run"
```

If a prior run exists and `--resume` was NOT specified, ask:
> "Found existing checkpoint at `./exp/{project}/{model_name}/last.ckpt`. Resume? (yes / no / abort)"

### 3. Show launch plan

Display:
```
Experiment:   {model_name}
Config:       {config_path}
Steps:        {steps}
GPU:          {gpu or "auto"}
Resume:       {resume path or "no"}
Output:       ./exp/{project}/{model_name}/
TensorBoard:  tensorboard --logdir ./exp/{project}/
```

### 4. Launch

```bash
python train.py --config {config_path} [--gpu {gpu}] [--resume {resume}]
```

Stream output live. After launch (or if running in background), print:
```
Monitor:    tensorboard --logdir ./exp/
Status:     /status
Results:    /results
```

### 5. Update research/log.md

Append:
```markdown
## {YYYY-MM-DD HH:MM} — Train: {model_name}
- Config: {config_path}
- Steps: {steps}
- Resume: {resume or "no"}
```
