# /iterate — Plan the Next Research Cycle

Review the full research history and propose the next round of hypotheses, ranked by expected value.

**Input:** $ARGUMENTS (optional — a specific question or constraint to focus the iteration, e.g. "focus on reducing overfitting" or "explore different architectures")

## Step-by-Step Instructions

### 1. Load research history

Read all available research artifacts:
- `research/log.md` — chronological record
- `research/hypotheses/*.md` — all past hypothesis reviews and their GO/REFINE/NO-GO status
- `research/experiments/*/analysis.md` — all analysis reports and findings

Summarize the history:
- How many hypotheses evaluated, how many ran as experiments
- Overall trend in val_loss across experiments
- Key findings identified so far
- Open questions not yet addressed

### 2. Spawn two synthesis agents IN PARALLEL

---

**Agent A — Research Trajectory Analyst**

Prompt:
> You are a research advisor reviewing an ML research trajectory. Here is the history: {history_summary}.
> Identify: (1) What is the strongest signal found so far — what clearly works or doesn't work? (2) What is the biggest open question that the existing experiments have NOT answered? (3) What is the most promising unexplored direction, given what's been learned? Return: strongest signal, biggest open question, top unexplored direction.

---

**Agent B — Hypothesis Generator**

Prompt:
> You are generating new research hypotheses based on prior findings. Research history: {history_summary}. Focus constraint (if any): "{arguments}".
> Generate exactly 3 new hypotheses. For each: (1) State the hypothesis clearly in one sentence. (2) Explain why the history motivates it. (3) Predict the expected outcome. (4) Rate expected value (1–5): novelty × feasibility × impact.
> Format each as: Hypothesis N | Motivation | Prediction | EV score.

---

### 3. Synthesize and rank

Combine Agent A and B outputs. Rank the 3 proposed hypotheses by expected value. For the top-ranked hypothesis, draft the one-liner ready to feed into `/hypothesis`.

### 4. Update research log

Append to `research/log.md`:
```markdown
## {YYYY-MM-DD} — Iteration Planning
- Experiments completed so far: {N}
- Strongest signal: {signal}
- Open question: {question}
- Proposed next hypotheses (ranked):
  1. {hypothesis 1}
  2. {hypothesis 2}
  3. {hypothesis 3}
```

### 5. Present to user

Display:
1. **Research status**: what's been learned
2. **Top 3 hypotheses** (ranked, with EV scores and motivation)
3. **Recommended next step**: the exact `/hypothesis` command to run for the top pick

Example closing:
```
Top pick → /hypothesis {top hypothesis text}
```
