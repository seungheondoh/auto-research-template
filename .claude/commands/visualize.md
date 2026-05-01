# /visualize — Generate Experiment Figures

Generate standard diagnostic plots for a training run.

**Input:** $ARGUMENTS  
(Experiment name like `ddpm`, or path like `exp/generative/ddpm`)

## Step-by-Step Instructions

### 1. Locate experiment directory

Resolve `{exp_dir}` the same way as `/analyze`: search `./exp/` for a matching directory if a bare name is given.

Create the output directory:
```bash
mkdir -p research/experiments/{model_name}/figures
```

### 2. Generate plots

Run the visualization script:
```bash
python scripts/plot_metrics.py \
    --exp_dir {exp_dir} \
    --out_dir research/experiments/{model_name}/figures
```

This produces:
- `loss_curve.png` — train_loss and val_loss vs steps
- `lr_schedule.png` — learning rate schedule
- `val_metrics.png` — all validation metrics over time
- `summary.png` — 2×2 dashboard of the above

### 3. Report outputs

List the generated files with their paths. If a file wasn't generated (metric not logged), note why.

### 4. Update analysis report

If `research/experiments/{model_name}/analysis.md` exists, append a `## Figures` section listing the generated plot paths.

### 5. Present to user

Show the list of generated figures. Suggest:
- Open in any image viewer
- `/iterate` if ready to plan the next experiment cycle
