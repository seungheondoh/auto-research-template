# /discuss — Collaborative Research Ideation Pipeline

The user's research idea is:

> $ARGUMENTS

Execute the following phases in order. **Pause for human input after Phase 1 and Phase 2** before continuing.

---

## Phase 1 — Understand & Restate

Before generating any hypotheses, restate your understanding of the research idea in this exact format:

---
**Domain:** [ML subfield — e.g., generative modeling, self-supervised learning, LLM alignment]
**Problem:** [What is currently broken or missing?]
**Proposed direction:** [What is the user's intuition or approach?]
**Template fit:** [Which `trainer_type` + `model_type` would apply, or what new trainer is needed?]

---

Then ask:

> Does this framing match your intent? Correct anything before I generate hypotheses — it's cheap to fix now. Reply "yes" or describe any corrections.

**Wait for the user's response before proceeding to Phase 2.**

---

## Phase 2 — Divergent Ideation (5 parallel sub-agents)

Spawn exactly **5 sub-agents in parallel** — send all 5 Agent calls in a single response turn so they run concurrently.

Each sub-agent receives:
- The original idea: `$ARGUMENTS`
- The restatement from Phase 1 (including any corrections the user made)
- Its assigned angle (see below)

Brief all 5 sub-agents with this shared context:

> "You are a research ideation agent. The project is a PyTorch Lightning template
> supporting three paradigms: lm (causal LM, multimodal), generative (vae, ddpm,
> flow_matching, drift), representation (contrastive, byol, mae).
> The user's idea is: $ARGUMENTS
> Your task: propose ONE concrete research hypothesis from your assigned angle.
> Return ONLY a raw JSON object — no markdown, no explanation."

Sub-agent angle assignments:

| Agent | Angle | Instruction |
|-------|-------|-------------|
| 1 | **Methodological** | Propose a new algorithm or architecture as the core contribution |
| 2 | **Application Transfer** | Propose applying the idea to a new domain or downstream task |
| 3 | **Efficiency & Scalability** | Propose making the approach faster, cheaper, or more parameter-efficient |
| 4 | **Theoretical** | Propose a theoretical insight, bound, or analysis as the contribution |
| 5 | **Data & Training** | Propose a novel data strategy, augmentation, or training recipe |

Each sub-agent must return exactly this JSON (no other text):

```json
{
  "Name": "kebab-case-slug-max-4-words",
  "Title": "Full descriptive research title",
  "Short Hypothesis": "One sentence: [method] achieves [outcome] by [mechanism].",
  "Angle": "methodological | application | efficiency | theoretical | data"
}
```

After collecting all 5 outputs, display them as a numbered summary table:

| # | Name | Angle | Short Hypothesis |
|---|------|-------|-----------------|
| 1 | ... | ... | ... |
| 2 | ... | ... | ... |
| 3 | ... | ... | ... |
| 4 | ... | ... | ... |
| 5 | ... | ... | ... |

Then ask:

> Which of these do you want me to enrich? You can:
> - Say "all" to enrich all five
> - List numbers to keep (e.g., "1, 3, 4")
> - Drop any that are out of scope, too expensive, or overlap with prior work you know
>
> Each hypothesis costs ~5 Semantic Scholar API calls to enrich.

**Wait for the user's response before proceeding to Phase 3.**

---

## Phase 3 — Evaluation & Enrichment (main agent)

Enrich only the hypotheses the user selected. For each one, run sequentially:

### 3a. Related Work
```bash
python scripts/semantic_scholar.py "<concise query for this hypothesis>" --limit 5
```
From the results, identify the 2–3 most relevant papers and the gap this hypothesis fills.
Write `Related Work` as: "Prior work X (YEAR) does Y. Our hypothesis differs by Z."

### 3b. Abstract
Write a 4-sentence mini-abstract:
1. **Motivation** — what problem exists and why it matters
2. **Method** — what we propose to do
3. **Key insight** — the mechanism or intuition that makes it work
4. **Contribution** — expected outcome or advance

### 3c. Experiments
Describe 2–3 concrete experiments:
- Which config file to start from (`config/<paradigm>/<file>.yaml`)
- Baselines to compare against
- Primary metric and what improvement would validate the hypothesis

### 3d. Risk Factors and Limitations
List 2–3 concrete risks:
- Technical failure modes
- Data or compute requirements that may not be met
- Scenarios where the hypothesis would be disproven

Assemble the complete object for each enriched hypothesis:

```json
{
  "Name": "...",
  "Title": "...",
  "Short Hypothesis": "...",
  "Related Work": "...",
  "Abstract": "...",
  "Experiments": "...",
  "Risk Factors and Limitations": "..."
}
```

---

## Phase 4 — Save

Save to `.research/discussions/YYYY-MM-DD-<slug>.json`:
- Use today's actual date for `YYYY-MM-DD`
- `<slug>` = first 3–4 words of the idea, lowercased, hyphenated

File contents must be a valid JSON array containing only the enriched hypotheses (not skipped ones):

```json
[
  {
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
  },
  ...
]
```

After saving:
1. Print the full JSON array inline
2. Print a one-line summary table
3. Print: `Saved → .research/discussions/<filename>.json`
4. Ask: "Which hypothesis do you want to turn into an experiment? I can create the branch and copy the config."
