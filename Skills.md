# Skills Quick Reference

## /discuss — Collaborative Research Ideation

```
/discuss <your research idea>
```

Pauses at two human decision points before spending compute on enrichment.

### Pipeline

```
Phase 1 — Understand & Restate (Claude)
  └─ States: domain / problem / direction / template fit
  └─ ★ WAIT — you review and correct before proceeding

Phase 2 — Divergent Ideation (5 parallel sub-agents)
  ├─ Agent 1: Methodological angle
  ├─ Agent 2: Application transfer angle
  ├─ Agent 3: Efficiency & scalability angle
  ├─ Agent 4: Theoretical angle
  └─ Agent 5: Data & training strategy angle
  └─ ★ WAIT — you filter hypotheses before enrichment

Phase 3 — Evaluation & Enrichment (Claude, per selected hypothesis)
  └─ python scripts/semantic_scholar.py "<query>" --limit 5
  └─ fills Related Work, Abstract, Experiments, Risk Factors

Phase 4 — Save
  └─ .research/discussions/YYYY-MM-DD-<slug>.json
  └─ git add runs automatically (PostToolUse hook)
```

### Your checkpoints

| Pause point | What to do | Why it matters |
|-------------|-----------|----------------|
| After Phase 1 | Correct domain / framing | Prevents 5 wrong hypotheses |
| After Phase 2 | Drop out-of-scope angles | Each dropped = 5 fewer API calls |
| After Phase 3 | Pick hypothesis to implement | Directs the next experiment branch |

### Output JSON schema (per hypothesis)

```json
{
  "Name": "kebab-case-slug",
  "Title": "Full descriptive research title",
  "Short Hypothesis": "One sentence: method + outcome + mechanism.",
  "Related Work": "Prior work X (YEAR) does Y. Gap: Z.",
  "Abstract": "4-sentence mini-abstract.",
  "Experiments": "Which configs, baselines, and metrics.",
  "Risk Factors and Limitations": "2-3 concrete risks."
}
```

### Examples

```
/discuss flow matching for protein structure prediction
/discuss LoRA rank sensitivity across model scales
/discuss self-supervised audio with masked spectrogram modeling
```

### Optional

Set `SEMANTIC_SCHOLAR_API_KEY` to raise API rate limits (100 req/5min → 1 req/sec).

---

## /train — Start Training

Asks which config to use, then runs `python train.py --config <path>`.

---

## /status — Training Status

Reads `./exp/` logs and reports latest metrics.

---

## /results — Experiment Results

Shows metrics from `./exp/` and lists recent JSON briefs in `.research/discussions/`.

---

## Hooks (automatic)

| Trigger | Action |
|---------|--------|
| Write to `.research/discussions/*.json` | `git add <file>` + commit reminder |

Configured in `.claude/settings.json`.
