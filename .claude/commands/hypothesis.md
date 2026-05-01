# /hypothesis — Research Hypothesis Review

Evaluate the following research hypothesis using a three-reviewer sub-agent panel.

**Hypothesis:** $ARGUMENTS

## Step-by-Step Instructions

### 1. Prepare output path

Get today's date (YYYY-MM-DD format). Generate a slug from the hypothesis (snake_case, max 5 words, English). The report will go to `research/hypotheses/YYYY-MM-DD_{slug}.md`. Create the directory if it doesn't exist.

### 2. Spawn three reviewer agents IN PARALLEL

Use the Agent tool to spawn all three at the same time. Each should receive the full hypothesis text and only the role below.

---

**Agent A — Theory Reviewer**

Prompt:
> You are a scientific peer reviewer. Evaluate this research hypothesis: "{hypothesis}".
> Assess: (1) Is it falsifiable and well-scoped? (2) What prior work directly supports or contradicts it — cite specific methods (e.g. DDPM, InfoNCE, BYOL, MAE, flow matching). (3) What are the main theoretical risks or assumptions that could make it fail? Rate scientific validity 1–5. Return exactly: rating, 3 strengths, 3 concerns.

---

**Agent B — Experimental Design Reviewer**

Prompt:
> You are an ML experiment design reviewer. Evaluate the experimental design implied by this hypothesis: "{hypothesis}".
> Assess: (1) What metrics would cleanly validate or refute it? (2) What confounds or sources of bias exist? (3) What ablations and baselines are essential? Rate design quality 1–5. Return exactly: rating, recommended metrics, ablation plan, confound warnings.

---

**Agent C — Feasibility Reviewer**

Prompt:
> You are reviewing whether this hypothesis can be tested using the auto-research-template codebase (PyTorch Lightning, trainer types: lm / generative / representation, model types: causal / vae / ddpm / flow_matching / drift / contrastive / byol / mae). Hypothesis: "{hypothesis}".
> Assess: (1) Which trainer_type and model_type best tests this? (2) What config fields need to change? (3) Estimate GPU-hours on a single A100 for a minimal proof-of-concept run. Rate implementation feasibility 1–5. Return exactly: rating, trainer_type, model_type, key config changes, compute estimate.

---

### 3. Synthesize

After all three agents return:
- Compute average score
- **GO** if avg ≥ 4.0, **REFINE** if 2.5–3.9, **NO-GO** if < 2.5
- If REFINE: list the 2–3 specific changes needed before proceeding
- If GO: state the recommended config command

### 4. Save report

Write `research/hypotheses/{date}_{slug}.md` using this exact structure:

```markdown
# Hypothesis Review: {slug}

**Date:** {YYYY-MM-DD}
**Status:** GO / REFINE / NO-GO
**Overall Score:** X.X / 5

## Hypothesis
{full hypothesis text}

## Theory Review — Score: X/5
{Agent A output}

## Design Review — Score: X/5
{Agent B output}

## Feasibility Review — Score: X/5
**Recommended:** trainer_type={} model_type={}
{Agent C output}

## Synthesis
{your synthesis: key agreements across reviewers, main risk, recommendation rationale}

## Action Items
{list of concrete next steps}

## Next Command
/experiment research/hypotheses/{date}_{slug}.md
```

### 5. Present to user

Display the synthesis section and action items. Tell the user the report was saved and suggest the next command.
