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
│   └── semantic_scholar.py
└── .research/
    └── discussions/
```

## Training Commands

```bash
python train.py --config config/lm/default.yaml
python train.py --config config/generative/ddpm.yaml
python train.py --config config/generative/flow_matching.yaml
python train.py --config config/generative/vae.yaml
python train.py --config config/representation/contrastive.yaml
python train.py --config config/representation/byol.yaml

python train.py --config config/generative/ddpm.yaml --gpu 0
python train.py --config config/generative/ddpm.yaml --resume ./exp/generative/ddpm/last.ckpt
tensorboard --logdir ./exp/
```

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

## Slash Commands

| Command | What it does |
|---------|-------------|
| `/discuss <idea>` | Collaborative ideation — pauses for human input after restatement and after hypothesis generation |
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
    discuss.md           # /discuss orchestration prompt
  hooks/
    on_write.sh          # auto-stages .research/discussions/*.md after Write
```

**Hook:** when Claude saves a file under `.research/discussions/`, `git add` runs automatically.

## Permissions

Pre-approved (no prompt needed):
- `python scripts/semantic_scholar.py *`
- `git add .research/discussions/*`
- `git commit *`
- `WebFetch https://api.semanticscholar.org/*`

Denied: `rm -rf`, `dd`, `mkfs`
