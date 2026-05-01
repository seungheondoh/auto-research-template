# Auto-Research Template

PyTorch Lightning research template — LM, generative, and representation learning.

## Quick Start

```bash
source scripts/setup_env.sh          # set SLACK_WEBHOOK_URL, OBSIDIAN_VAULT_PATH
uv venv .venv && source .venv/bin/activate
uv pip install -e .
python train.py --config config/generative/ddpm.yaml
```

## Trainer Registry

| `trainer_type` | Paradigm | Backbone |
|----------------|----------|----------|
| `pretrain` | Causal LM pretraining | Qwen2.5-0.5B |
| `sft` | Supervised fine-tuning (LoRA) | Qwen2.5-0.5B |
| `posttrain` | DPO preference optimization | Qwen2.5-0.5B |
| `rl` | GRPO reward RL | Qwen2.5-0.5B |
| `infonce` | InfoNCE / NT-Xent contrastive | DiT (ViTEncoder) |
| `jepa` | I-JEPA latent prediction | DiT (ViTEncoder + EMA) |
| `diffusion` | DDPM ε/x₀/v prediction | DiT |
| `flow_matching` | OT-CFM velocity field | DiT |
| `drift` | Stochastic interpolant | DiT |
| `vae` | β-VAE | ViTEncoder + ViTDecoder |
| `vqvae` | VQ-VAE discrete tokens | ViTEncoder + VQ + ViTDecoder |

## Project Layout

```
auto-research-template/
├── train.py
├── config/
│   ├── lm/{pretrain,sft,posttrain,rl}.yaml
│   ├── representation/{infonce,jepa}.yaml
│   ├── generative/{diffusion,flow_matching,drift}.yaml
│   └── vae/{vae,vqvae}.yaml
├── src/
│   ├── trainers/
│   │   ├── base_trainer.py           # AdamW + LinearWarmupCosineDecay
│   │   ├── lm/                       # pretrain, sft, posttrain (DPO), rl (GRPO)
│   │   ├── representation/           # infonce, jepa
│   │   ├── generative/               # diffusion, flow_matching, drift
│   │   └── vae/                      # vae, vqvae
│   ├── modules/                      # shared components
│   │   ├── dit.py                    # DiT denoiser, ViTEncoder, ViTDecoder
│   │   ├── lm.py                     # Qwen2.5-0.5B builder + LoRA
│   │   ├── ema.py                    # EMA wrapper
│   │   ├── noise_scheduler.py        # DDPM schedule + flow interpolation
│   │   ├── quantizer.py              # VectorQuantizer (VQ-VAE)
│   │   └── losses.py                 # InfoNCE, ELBO, VQ loss
│   ├── data/                         # JsonlDataset, ImageFolderDataset, ResearchDataModule
│   ├── callbacks/                    # SlackNotificationCallback, ObsidianLogCallback
│   └── utils/                        # instantiate_from_config, LinearWarmupCosineDecay
├── scripts/
│   ├── setup_env.sh
│   ├── semantic_scholar.py
│   ├── orchestrator.py           # multi-agent sweep report (Haiku × N → Sonnet)
│   ├── run_sweep.sh              # sequential sweep runner
│   ├── read_metrics.py           # parse metrics.csv → text + JSON summary
│   └── plot_metrics.py           # generate loss_curve/lr_schedule/summary.png
├── sweep.config.yaml             # experiment ablation manifest
├── research/
│   ├── hypotheses/               # /hypothesis output (YYYY-MM-DD_slug.md)
│   ├── experiments/              # /analyze + /visualize output per run
│   └── log.md                    # running research journal
└── .research/
    └── discussions/              # /discuss output
```

## Training Commands

```bash
# Single run
python train.py --config config/generative/ddpm.yaml
python train.py --config config/generative/ddpm.yaml --gpu 0
python train.py --config config/generative/ddpm.yaml --resume ./exp/generative/ddpm/last.ckpt
tensorboard --logdir ./exp/

# Sweep (sequential, all experiments in sweep.config.yaml)
bash scripts/run_sweep.sh
bash scripts/run_sweep.sh --gpu 0
bash scripts/run_sweep.sh --only flow_matching

# Multi-agent comparative report (Haiku sub-agents → Sonnet synthesis)
python scripts/orchestrator.py
python scripts/orchestrator.py --only ddpm
```

## Sweep + Orchestrator Workflow

```
sweep.config.yaml          declare experiments + hypotheses
        │
bash scripts/run_sweep.sh  run all sequentially
        │
python scripts/orchestrator.py
        │
        ├─ Sub-agent (Haiku) × N experiments ─── parallel analysis
        └─ Main agent (Sonnet) ────────────────── comparative report
                                                  → research/experiments/{sweep}_report.md
                                                  → $OBSIDIAN_VAULT_PATH/experiments/
```

**To add a new ablation:** copy an entry in `sweep.config.yaml`, set `id`, `config`, and `hypothesis`. Then re-run `run_sweep.sh --only {new_id}` and `orchestrator.py`.

## Config Schema

```yaml
project: "generative"
model_name: ddpm
steps: 100000
logdir: ./exp
expdir: ${logdir}/${project}/${model_name}

model:
  target: src.trainers.generative_trainer.GenerativeTrainer
  params:
    cfg:
      trainer_type: generative
      model_type: ddpm

data:
  target: src.data.datamodule.ResearchDataModule
  params:
    cfg:
      data_dir: ./data
      batch_size: 16

lightning:
  slack_notify_every_n_epochs: 1
  obsidian_log_every_n_epochs: 1
  trainer:
    accelerator: gpu
    strategy: ddp
    max_steps: ${steps}
```

## Extending

**New experiment:** copy an existing config, change `model_name` + hyperparams.

**New trainer:**
```bash
vi src/trainers/my_trainer.py        # subclass BaseTrainer
vi src/trainers/__init__.py          # add to dispatch dict
cp config/generative/ddpm.yaml config/my_paradigm/default.yaml
```

**Custom dataset:** subclass `ResearchDataModule` and override `_build_datasets()`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SLACK_WEBHOOK_URL` | Optional | Slack epoch notifications |
| `OBSIDIAN_VAULT_PATH` | Optional | Obsidian vault git repo path |
| `ANTHROPIC_API_KEY` | Optional | Multi-agent scripts |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | Higher rate limit for `/discuss` |
| `CUDA_VISIBLE_DEVICES` | Optional | GPU selection (or use `--gpu`) |

## Research Loop

Agent-assisted iterative research cycle. Each phase uses parallel sub-agents to review or analyze, then feeds results back to the main agent.

```
/hypothesis → 3-Agent Review → /experiment → /analyze (3-Agent) → /visualize → /iterate
      ▲                                                                              │
      └──────────────────────────────────────────────────────────────────────────────┘
```

### Commands

| Command | Input | What happens |
|---------|-------|-------------|
| `/hypothesis <text>` | Hypothesis statement | 3 reviewer agents evaluate in parallel → GO/REFINE/NO-GO saved to `research/hypotheses/` |
| `/experiment <file or config>` | Hypothesis `.md` or config path | Reads recommended config, launches training, logs to `research/log.md` |
| `/analyze <exp-name>` | Experiment name or path | 3 analysis agents review results → `research/experiments/{name}/analysis.md` |
| `/visualize <exp-name>` | Experiment name | Generates loss curves + metric plots → `research/experiments/{name}/figures/` |
| `/iterate [focus]` | Optional focus constraint | Reviews all past cycles, proposes top-3 next hypotheses by expected value |
| `/discuss <idea>` | Research idea | Collaborative ideation — pauses for human input (existing command) |

### `/hypothesis` — 3 Parallel Reviewer Agents

- **Theory Agent**: scientific validity, falsifiability, prior art — score 1–5
- **Design Agent**: metrics, confounds, ablations, baselines — score 1–5
- **Feasibility Agent**: trainer_type, model_type, config changes, GPU estimate — score 1–5

Average → **GO** (≥ 4.0) / **REFINE** (2.5–3.9) / **NO-GO** (< 2.5)

### `/analyze` — 3 Parallel Analysis Agents

- **Stats Agent**: convergence, train/val gap, best checkpoint step
- **Ablation Agent**: top drivers, 3 ranked next ablations
- **Comparison Agent**: hypothesis confirmed / partial / refuted + mechanistic insight

### `/iterate` — 2 Parallel Synthesis Agents

- **Trajectory Analyst**: strongest signal, biggest open question, top unexplored direction
- **Hypothesis Generator**: 3 new hypotheses with motivation, prediction, EV score (novelty × feasibility × impact)

### Research Directory

```
research/
├── hypotheses/                    # /hypothesis output
│   └── YYYY-MM-DD_{slug}.md
├── experiments/                   # /analyze + /visualize output
│   └── {model_name}/
│       ├── analysis.md
│       └── figures/               # loss_curve.png, lr_schedule.png, summary.png
└── log.md                         # appended by /experiment, /analyze, /iterate
```

Research files are auto-staged by the `PostToolUse[Write]` hook:
```bash
git commit -m "research: <summary>"
```

### Supporting Scripts

```bash
python scripts/read_metrics.py exp/generative/ddpm          # parse + summarize metrics
python scripts/plot_metrics.py --exp_dir exp/generative/ddpm --out_dir research/experiments/ddpm/figures
```

## Slash Commands

| Command | What it does |
|---------|-------------|
| `/discuss <idea>` | Collaborative ideation — pauses for human input after restatement and after hypothesis generation |
| `/hypothesis <text>` | 3-agent hypothesis review → GO/REFINE/NO-GO |
| `/experiment <file>` | Launch training from hypothesis or config |
| `/analyze <exp-name>` | 3-agent result analysis |
| `/visualize <exp-name>` | Generate diagnostic plots |
| `/iterate [focus]` | Plan next research cycle |
| `/train` | Start a training run |
| `/status` | Check current training status |
| `/results` | Show latest experiment metrics |

### /discuss

Four-phase collaborative pipeline. **Pauses twice for human input** before spending compute:

1. **Phase 1** — Restates the research idea; waits for the human to confirm or correct the framing
2. **Phase 2** — Spawns 5 parallel sub-agents across different research angles; waits for the human to filter hypotheses before enrichment
3. **Phase 3** — Runs `python scripts/semantic_scholar.py "<query>"` per selected hypothesis, fills Related Work / Abstract / Experiments / Risk Factors
4. **Phase 4** — Saves enriched hypotheses to `.research/discussions/YYYY-MM-DD-<slug>.json`

```
/discuss flow matching for protein structure prediction
/discuss self-supervised audio with masked spectrogram modeling
```

## Harness Layout

```
.claude/
  settings.json          # project permissions + hooks (committed)
  settings.local.json    # personal overrides (gitignored)
  commands/
    discuss.md           # /discuss — collaborative ideation (Semantic Scholar)
    hypothesis.md        # /hypothesis — 3-agent parallel review → GO/REFINE/NO-GO
    experiment.md        # /experiment — launch from hypothesis .md or config path
    analyze.md           # /analyze — 3-agent parallel result analysis
    visualize.md         # /visualize — generate diagnostic plots via plot_metrics.py
    iterate.md           # /iterate — 2-agent synthesis → ranked next hypotheses
    train.md             # /train — launch with pre-flight checks
    status.md            # /status — live training status
    results.md           # /results — comparison table + orchestrator trigger
  hooks/
    on_write.sh          # auto-stages .research/discussions/*.md and research/*.md
```

**Hook:** any file saved under `.research/discussions/` or `research/` is auto-`git add`ed.

## Permissions

Broad file and bash permissions are pre-approved (see `.claude/settings.json`).
Key allowlist: `Read`, `Edit`, `Write`, `Bash(python *)`, `Bash(git *)`, `Bash(ls/find/grep/cat/tail/head/mkdir/cp/mv *)`, `Bash(tensorboard *)`.

Denied: `rm -rf`, `rm -fr`, `sudo rm`, `dd`, `mkfs`, `shred`.
