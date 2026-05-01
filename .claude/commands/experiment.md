# /experiment — Run a Training Experiment

Launch a training experiment linked to a hypothesis review.

**Input:** $ARGUMENTS  
(Either a path to a hypothesis `.md` file, or a `trainer_type/model_type` config path like `config/generative/ddpm.yaml`)

## Step-by-Step Instructions

### 1. Resolve the config

**If input is a hypothesis `.md` file:**
- Read the file and extract the "Feasibility Review" section
- Find the `trainer_type` and `model_type` fields
- Map them to a config path: `config/{trainer_type}/{model_type}.yaml`
- If that config doesn't exist, use the closest available config as a base
- Copy it to `config/{trainer_type}/{slug}.yaml` where slug comes from the hypothesis filename

**If input is already a config path:**
- Use it directly

### 2. Show experiment plan

Display:
- Config path to be used
- Key hyperparameters (model_type, batch_size, steps, lr)
- Linked hypothesis (if applicable)
- Estimated GPU requirement

Ask for confirmation before launching: "Ready to launch? (yes / change config / abort)"

### 3. Launch

Run:
```bash
python train.py --config {config_path}
```

Stream output. After launch, remind the user they can:
- Monitor: `tensorboard --logdir ./exp/`
- Resume: `python train.py --config {config_path} --resume ./exp/{project}/{model_name}/last.ckpt`
- Analyze when done: `/analyze {model_name}`

### 4. Update research log

Append to `research/log.md`:
```markdown
## {YYYY-MM-DD} — Experiment: {model_name}
- Config: {config_path}
- Hypothesis: {hypothesis_file or "manual"}
- Status: launched
- Command: python train.py --config {config_path}
```

Create `research/log.md` with a header if it doesn't exist yet:
```markdown
# Research Log

Chronological record of hypotheses, experiments, and findings.
```
