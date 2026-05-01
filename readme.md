# Auto-Research Template

A PyTorch Lightning research template built for **human-AI collaborative research** — Claude handles literature search, hypothesis generation, and experiment scaffolding; you direct the science.

| Paradigm | `trainer_type` | `model_type` options |
|----------|---------------|----------------------|
| Language Model | `lm` | causal, multimodal (prefix / cross_attn / adaLN) |
| Generative | `generative` | `vae`, `ddpm`, `flow_matching`, `drift` |
| Representation | `representation` | `contrastive`, `byol`, `mae` |

---

## Collaboration Model

This template structures research as a back-and-forth between you and Claude Code:

| You bring | Claude brings |
|-----------|--------------|
| Research intuition and domain taste | Literature search (Semantic Scholar) |
| Judgment on what to pursue | Structured hypothesis generation |
| Experiment decisions | Config scaffolding and code changes |
| Interpretation of results | Metric summaries and TensorBoard links |

The `/discuss` command is the primary collaboration interface. It pauses at two decision points — **after restating your idea** and **after generating hypotheses** — so you stay in control of the scientific direction before any compute is spent.

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

After each validation epoch, the `SlackNotificationCallback` posts `val_loss`, `train_loss`, and a generated sample to that channel.

#### Obsidian (optional — experiment notes)

Writes `.md` notes to a git repo that syncs with the [Obsidian](https://obsidian.md/) app via the [Obsidian Git](https://github.com/denolehov/obsidian-git) plugin.

```bash
git clone https://github.com/<you>/obsidian-sync /workspace/obsidian-sync
export OBSIDIAN_VAULT_PATH="/workspace/obsidian-sync"
```

The `ObsidianLogCallback` commits and pushes `experiments/{model_name}.md` with metrics and an ASCII loss curve after every validation epoch.

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

## Part 2 — Research Discussion (`/discuss`)

Use `/discuss` when you have a rough idea and want structured hypotheses + a Semantic Scholar novelty check before committing to an experiment.

### Collaborative pipeline

```
You: /discuss <your research idea>
              │
              ▼
Phase 1 — Understand & Restate  (Claude)
  Claude states: domain / problem / direction / template fit.
  ──────────────────────────────────────────────────────
  ★ YOU REVIEW: Is the framing right? Correct it before
    hypotheses are generated — cheap to fix here.
  ──────────────────────────────────────────────────────
              │  (your go-ahead)
              ▼
Phase 2 — Divergent Ideation  (5 parallel sub-agents)
  ┌─ Agent 1: Methodological angle
  ├─ Agent 2: Application transfer angle
  ├─ Agent 3: Efficiency & scalability angle
  ├─ Agent 4: Theoretical angle
  └─ Agent 5: Data & training strategy angle
  ──────────────────────────────────────────────────────
  ★ YOU REVIEW: Drop angles outside your scope/budget.
    Each dropped hypothesis saves 5 API calls.
  ──────────────────────────────────────────────────────
              │  (your selection)
              ▼
Phase 3 — Evaluation & Enrichment  (Claude, per hypothesis)
  python scripts/semantic_scholar.py "<query>" --limit 5
  → Related Work, Abstract, Experiments, Risk Factors
              │
              ▼
Phase 4 — Save
  .research/discussions/YYYY-MM-DD-<slug>.json
  JSON array — git-staged automatically.
```

### Your decision points

**After Phase 1 — correct the framing**

Claude restates its understanding before generating anything. If the domain or direction is off, say so now.

```
Claude: "Domain: protein structure prediction, direction: flow matching..."
You:    "Close — apply it to side-chain packing, not backbone folding."
```

**After Phase 2 — filter hypotheses**

Five hypotheses are shown before any Semantic Scholar queries run. Drop any that clearly overlap with work you know, don't fit your data, or exceed your compute budget.

```
Claude: shows 5 hypotheses
You:    "Agent 3 (efficiency) is out of scope. Skip it and enrich the other four."
```

**After Phase 3 — pick what to run**

All enriched hypotheses are saved. Tell Claude which one to turn into an experiment — it will copy the right config and create a git branch.

```
You: "H2 (application transfer) looks most actionable. Let's start there."
Claude: sets up experiment/protein-side-chain branch and edits the config.
```

### Output schema (per hypothesis)

```json
{
  "Name": "kebab-case-slug",
  "Title": "Full descriptive research title",
  "Short Hypothesis": "One sentence: method + outcome + mechanism.",
  "Related Work": "Prior work X (YEAR) does Y. Gap: Z.",
  "Abstract": "4-sentence mini-abstract.",
  "Experiments": "Which configs, baselines, and metrics to run.",
  "Risk Factors and Limitations": "2-3 concrete risks."
}
```

### Examples

```
/discuss flow matching for protein structure prediction
/discuss self-supervised audio with masked spectrogram modeling
/discuss LoRA rank sensitivity across model scales
```

---

## Part 3 — Git-Based Experiment Branching

Use one branch per hypothesis so you can run experiments in parallel, compare results cleanly, and merge only what works.

### Branch convention

```
main                        ← stable, working codebase
experiment/<slug>           ← one branch per hypothesis

Examples:
  experiment/flow-matching-protein
  experiment/masked-spectrogram-ssl
  experiment/lora-rank-sensitivity
```

### Workflow

```bash
# 1. After /discuss, create a branch for the chosen hypothesis
git checkout -b experiment/flow-matching-protein

# 2. Copy and edit the relevant config
cp config/generative/flow_matching.yaml config/generative/protein_flow.yaml
# set model_name, hyperparams, data_dir

# 3. Train
python train.py --config config/generative/protein_flow.yaml

# 4. Check results
tensorboard --logdir ./exp/
/results

# 5. If promising, merge back
git checkout main
git merge experiment/flow-matching-protein

# 6. If not, leave or delete
git branch -d experiment/flow-matching-protein
```

### Running multiple hypotheses in parallel (git worktree)

`git worktree` lets you check out two branches side by side in separate directories — no stashing, no branch switching.

```bash
# Add a worktree for hypothesis 2 while hypothesis 1 runs in the main dir
git worktree add ../exp-masked-spectrogram experiment/masked-spectrogram-ssl

# Train hypothesis 1 (GPU 0)
python train.py --config config/representation/contrastive.yaml --gpu 0 &

# Train hypothesis 2 in the worktree (GPU 1)
cd ../exp-masked-spectrogram
python train.py --config config/representation/byol.yaml --gpu 1 &

# Compare both in TensorBoard
tensorboard --logdir_spec \
  h1:./exp/representation/contrastive,\
  h2:../exp-masked-spectrogram/exp/representation/byol
```

### Discussion briefs follow the branch

When Claude saves a `.research/discussions/` file, it is automatically staged by the post-write hook. Commit it on the experiment branch so the brief stays with the code that implements it.

```bash
git commit -m "discuss: flow matching for protein structure"
git checkout -b experiment/flow-matching-protein
```

---

## Part 4 — Training Reference

```bash
# All paradigms
python train.py --config config/lm/default.yaml
python train.py --config config/generative/ddpm.yaml
python train.py --config config/generative/flow_matching.yaml
python train.py --config config/generative/drift.yaml
python train.py --config config/generative/vae.yaml
python train.py --config config/representation/contrastive.yaml
python train.py --config config/representation/byol.yaml

# Options
python train.py --config <path> --gpu 0
python train.py --config <path> --resume ./exp/<project>/<name>/last.ckpt
tensorboard --logdir ./exp/
```

### New experiment from scratch

```bash
# 1. Discuss with Claude and pick a hypothesis
/discuss <idea>

# 2. Branch
git checkout -b experiment/<slug>

# 3. Copy nearest config and adjust
cp config/generative/ddpm.yaml config/generative/my_exp.yaml
# set: model_name, model_type, hyperparams, data_dir

# 4. Train
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
│   └── discussions/       # briefs saved by /discuss (auto-staged)
└── .claude/
    ├── settings.json      # permissions + hooks
    ├── commands/
    │   └── discuss.md     # /discuss orchestration
    └── hooks/
        └── on_write.sh    # auto-stages discussion files
```
