# Auto-Research Template

A PyTorch Lightning research template for **human-AI collaborative research** — Claude handles literature search, hypothesis generation, and experiment scaffolding; you direct the science.

| Paradigm | `trainer_type` | `model_type` options |
|----------|---------------|----------------------|
| Language Model | `lm` | `causal`, `multimodal` (prefix / cross_attn / adaLN) |
| Generative | `generative` | `vae`, `ddpm`, `flow_matching`, `drift` |
| Representation | `representation` | `contrastive`, `byol`, `mae` |

---

## How It Works

Research here is a loop, not a pipeline. Each cycle moves from rough idea → structured hypotheses → running experiment → new questions.

```
┌─────────────────────────────────────────────────────────────────┐
│                     RESEARCH LOOP                               │
│                                                                 │
│   You have an idea                                              │
│        │                                                        │
│        ▼                                                        │
│   /discuss <idea>          ← Claude + Semantic Scholar          │
│        │                                                        │
│        ├─ Phase 1: Claude restates your idea → you correct it   │
│        ├─ Phase 2: 5 hypotheses generated → you filter them     │
│        ├─ Phase 3: Claude enriches selected → related work,     │
│        │           abstract, experiments, risk factors          │
│        └─ Phase 4: saved to .research/discussions/              │
│        │                                                        │
│        ▼                                                        │
│   Pick a hypothesis                                             │
│        │                                                        │
│        ▼                                                        │
│   git checkout -b experiment/<slug>                             │
│   python train.py --config config/...                           │
│   tensorboard --logdir ./exp/                                   │
│        │                                                        │
│        ▼                                                        │
│   Review results → /discuss again with what you learned  ◄──┐  │
│        │                                                     │  │
│        └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

| You bring | Claude brings |
|-----------|--------------|
| Research intuition and domain taste | Literature search (Semantic Scholar) |
| Judgment on what to pursue | Structured hypothesis generation |
| Experiment decisions | Config scaffolding and code changes |
| Interpretation of results | Metric summaries and TensorBoard links |

---

## Part 1 — Setup

### 1.1 Install

```bash
git clone <this-repo> && cd auto-research-template
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

> Requires Python ≥ 3.10 and at least one CUDA GPU. Replace `uv` with `pip` if preferred.

### 1.2 API Keys

You need at least **Anthropic** to run `/discuss`. The rest are optional.

#### Anthropic (required for `/discuss`)

1. Go to [console.anthropic.com](https://console.anthropic.com) → **API Keys** → **Create Key**
2. Copy the key (starts with `sk-ant-`)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

#### Semantic Scholar (optional — raises `/discuss` rate limit)

Without a key: 100 requests / 5 minutes (usually enough for one `/discuss` session).
With a key: 1 request / second.

1. Go to [api.semanticscholar.org](https://api.semanticscholar.org/) → **Get API Key**
2. Fill out the form (instant approval for academic use)

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-key-here"
```

#### Slack (optional — epoch notifications)

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From scratch**
2. **Incoming Webhooks** → toggle on → **Add New Webhook to Workspace** → pick a channel
3. Copy the webhook URL

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T.../B.../xxx"
```

After each validation epoch, `SlackNotificationCallback` posts `val_loss`, `train_loss`, and a generated sample to that channel.

#### Obsidian (optional — experiment notes)

Writes `.md` notes to a git repo that syncs with [Obsidian](https://obsidian.md/) via the [Obsidian Git](https://github.com/denolehov/obsidian-git) plugin.

```bash
git clone https://github.com/<you>/obsidian-sync /workspace/obsidian-sync
export OBSIDIAN_VAULT_PATH="/workspace/obsidian-sync"
```

`ObsidianLogCallback` commits and pushes `experiments/{model_name}.md` with metrics and an ASCII loss curve after every validation epoch.

#### All at once

Edit `scripts/setup_env.sh` with your keys, then:

```bash
source scripts/setup_env.sh
```

### 1.3 Start Claude Code

```bash
claude
```

Permissions, hooks, and slash commands are pre-configured in `.claude/settings.json`.

---

## Part 2 — Your First Research Cycle (end to end)

This walks through one full loop from idea to running experiment. Follow this the first time.

### Step 1: Discuss your idea

```
/discuss flow matching for protein structure prediction
```

Claude runs a 4-phase pipeline. **It pauses twice for your input** before spending compute.

```
Phase 1 — Understand & Restate
  Claude states its understanding of your idea.
  ──────────────────────────────────────────────────────
  ★ YOU REVIEW: Is the domain and direction right?
    Correct it here — this is cheap to fix.
  ──────────────────────────────────────────────────────
  Reply: "yes" or describe corrections.

Phase 2 — Divergent Ideation  (5 parallel sub-agents)
  ┌─ Agent 1: Methodological angle
  ├─ Agent 2: Application transfer angle
  ├─ Agent 3: Efficiency & scalability angle
  ├─ Agent 4: Theoretical angle
  └─ Agent 5: Data & training strategy angle
  ──────────────────────────────────────────────────────
  ★ YOU REVIEW: Claude shows 5 short hypotheses.
    Drop any that are out of scope or over budget.
    Each dropped hypothesis = 5 fewer API calls.
  ──────────────────────────────────────────────────────
  Reply: "1, 3, 4" or "all"

Phase 3 — Enrichment  (per selected hypothesis)
  Runs semantic_scholar.py, fills related work,
  abstract, experiments, risk factors.

Phase 4 — Save
  .research/discussions/YYYY-MM-DD-<slug>.json
  Auto-staged by git hook.
```

**Example correction at Phase 1:**
```
Claude: "Domain: protein backbone folding, direction: flow matching..."
You:    "Close — target side-chain packing, not backbone."
```

**Example filtering at Phase 2:**
```
Claude: shows 5 hypotheses
You:    "Agent 3 (efficiency) is out of scope. Enrich 1, 2, 4."
```

### Step 2: Pick a hypothesis and branch

After Phase 3, Claude asks which hypothesis to turn into an experiment. It will create the branch and copy the config.

```bash
# Claude does this, or you can do it manually:
git checkout -b experiment/flow-matching-protein
cp config/generative/flow_matching.yaml config/generative/protein_flow.yaml
```

Commit the discussion brief on the branch so the scientific rationale stays with the code:

```bash
git add .research/discussions/2026-05-02-flow-matching-protein.json
git commit -m "discuss: flow matching for protein side-chain packing"
```

### Step 3: Train

Edit the config you copied (`config/generative/protein_flow.yaml`) to set `model_name`, `data_dir`, and any hyperparams.

```bash
python train.py --config config/generative/protein_flow.yaml
tensorboard --logdir ./exp/
```

Check progress:
```
/status    ← current training metrics
/results   ← latest val_loss and experiment list
```

### Step 4: Interpret and iterate

When training finishes (or when you have early results), come back to `/discuss` with what you learned:

```
/discuss flow matching for protein side-chain packing —
  our OT-CFM baseline hits 1.2Å RMSD on CASP14 but struggles
  on long loops. ideas for improving loop modeling?
```

This starts a new cycle. The new brief gets saved alongside the old one in `.research/discussions/`, building a record of your research thread.

---

## Part 3 — Running Hypotheses in Parallel

If you have multiple hypotheses worth testing at once, use `git worktree` to run them side by side without switching branches.

```bash
# Hypothesis 1 is already in the main working directory
# Add a worktree for hypothesis 2
git worktree add ../exp-masked-spectrogram experiment/masked-spectrogram-ssl

# GPU 0: hypothesis 1
python train.py --config config/representation/contrastive.yaml --gpu 0 &

# GPU 1: hypothesis 2 (in the worktree)
cd ../exp-masked-spectrogram
python train.py --config config/representation/byol.yaml --gpu 1 &

# Compare both in TensorBoard
tensorboard --logdir_spec \
  h1:./exp/representation/contrastive,\
  h2:../exp-masked-spectrogram/exp/representation/byol
```

When results are in:
```bash
# Merge what works
git checkout main
git merge experiment/masked-spectrogram-ssl

# Delete what doesn't
git branch -d experiment/contrastive-baseline
git worktree remove ../exp-masked-spectrogram
```

---

## Part 4 — Training Reference

```bash
# Language models
python train.py --config config/lm/default.yaml

# Generative
python train.py --config config/generative/ddpm.yaml
python train.py --config config/generative/flow_matching.yaml
python train.py --config config/generative/drift.yaml
python train.py --config config/generative/vae.yaml

# Representation
python train.py --config config/representation/contrastive.yaml
python train.py --config config/representation/byol.yaml

# Options
python train.py --config <path> --gpu 0
python train.py --config <path> --resume ./exp/<project>/<name>/last.ckpt
tensorboard --logdir ./exp/
```

### Config schema

```yaml
project: "generative"
model_name: my_exp
steps: 100000
logdir: ./exp
expdir: ${logdir}/${project}/${model_name}

model:
  target: src.trainers.generative_trainer.GenerativeTrainer
  params:
    cfg:
      trainer_type: generative
      model_type: flow_matching

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

---

## Part 5 — Extending the Template

### New experiment

Copy the nearest config and adjust `model_name` and hyperparams — no code changes needed.

```bash
cp config/generative/ddpm.yaml config/generative/my_exp.yaml
# edit model_name, model_type, data_dir, hyperparams
python train.py --config config/generative/my_exp.yaml
```

### New trainer type

```bash
vi src/trainers/my_trainer.py        # subclass BaseTrainer
vi src/trainers/__init__.py          # register in dispatch dict
cp config/generative/ddpm.yaml config/my_paradigm/default.yaml
# set target: src.trainers.my_trainer.MyTrainer
python train.py --config config/my_paradigm/default.yaml
```

### Custom dataset

Subclass `ResearchDataModule` and override `_build_datasets()` in `src/data/`.

---

## Slash Commands

| Command | What it does |
|---------|-------------|
| `/discuss <idea>` | Research ideation — 5 hypotheses, Semantic Scholar enrichment, saved to `.research/discussions/` |
| `/train` | Start a training run (prompts for config) |
| `/status` | Check current training metrics |
| `/results` | Show latest experiment results and discussion briefs |

---

## Project Layout

```
auto-research-template/
├── train.py
├── config/
│   ├── lm/default.yaml
│   ├── generative/{ddpm,vae,flow_matching,drift}.yaml
│   └── representation/{contrastive,byol}.yaml
├── src/
│   ├── trainers/          # base + lm + generative + representation
│   ├── data/              # JsonlDataset, ImageFolderDataset, ResearchDataModule
│   ├── callbacks/         # SlackNotificationCallback, ObsidianLogCallback
│   └── utils/             # instantiate_from_config, LinearWarmupCosineDecay
├── scripts/
│   ├── setup_env.sh
│   └── semantic_scholar.py
├── .research/
│   └── discussions/       # briefs saved by /discuss (auto-staged by hook)
└── .claude/
    ├── settings.json      # permissions + hooks
    ├── commands/
    │   └── discuss.md     # /discuss orchestration
    └── hooks/
        └── on_write.sh    # auto-stages discussion files after Write
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for `/discuss` | Claude API access |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | 1 req/sec instead of 100 req/5min |
| `SLACK_WEBHOOK_URL` | Optional | Epoch notifications |
| `OBSIDIAN_VAULT_PATH` | Optional | Auto-sync experiment notes |
| `CUDA_VISIBLE_DEVICES` | Optional | GPU selection (or use `--gpu`) |
