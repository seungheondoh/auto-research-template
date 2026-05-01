#!/usr/bin/env bash
# Run all experiments defined in sweep.config.yaml sequentially, then generate a report.
#
# Usage:
#   bash scripts/run_sweep.sh                      # all experiments, auto GPU
#   bash scripts/run_sweep.sh --gpu 0              # pin to GPU 0
#   bash scripts/run_sweep.sh --only flow_matching  # run one experiment only
#   bash scripts/run_sweep.sh --sweep my_sweep.yaml # use a different sweep config

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Parse arguments ───────────────────────────────────────────────────────────
GPU_ARG=""
ONLY=""
SWEEP_FILE="sweep.config.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)    GPU_ARG="--gpu $2";    shift 2 ;;
        --only)   ONLY="$2";             shift 2 ;;
        --sweep)  SWEEP_FILE="$2";       shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Environment ───────────────────────────────────────────────────────────────
if [[ -f scripts/setup_env.sh ]]; then
    source scripts/setup_env.sh
fi

if [[ ! -f "$SWEEP_FILE" ]]; then
    echo "Sweep config not found: $SWEEP_FILE"
    echo "Create one or run: cp sweep.config.yaml $SWEEP_FILE"
    exit 1
fi

# ── Read sweep config ─────────────────────────────────────────────────────────
SWEEP_NAME=$(python3 -c "import yaml; c=yaml.safe_load(open('$SWEEP_FILE')); print(c.get('name', 'sweep'))")
SWEEP_STEPS=$(python3 -c "import yaml; c=yaml.safe_load(open('$SWEEP_FILE')); print(c.get('steps', '?'))")

mapfile -t EXP_IDS < <(python3 -c "
import yaml
c = yaml.safe_load(open('$SWEEP_FILE'))
for e in c.get('experiments', []):
    print(e.get('id', e.get('model_name', 'unknown')))
")

mapfile -t CONFIGS < <(python3 -c "
import yaml
c = yaml.safe_load(open('$SWEEP_FILE'))
for e in c.get('experiments', []):
    print(e.get('config', ''))
")

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
printf "║  Sweep: %-49s ║\n" "${SWEEP_NAME}"
printf "║  Steps: %-49s ║\n" "${SWEEP_STEPS}"
printf "║  Experiments: %-43s ║\n" "${EXP_IDS[*]}"
[[ -n "$ONLY" ]] && printf "║  Filter: %-48s ║\n" "--only ${ONLY}"
[[ -n "$GPU_ARG" ]] && printf "║  GPU: %-51s ║\n" "${GPU_ARG}"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Run experiments ───────────────────────────────────────────────────────────
FAILED=()

for i in "${!EXP_IDS[@]}"; do
    exp_id="${EXP_IDS[$i]}"
    cfg="${CONFIGS[$i]}"

    if [[ -n "$ONLY" && "$exp_id" != "$ONLY" ]]; then
        echo ""
        echo "  [skip] $exp_id (--only $ONLY)"
        continue
    fi

    if [[ -z "$cfg" ]]; then
        echo ""
        echo "  [skip] $exp_id — no config path in sweep config"
        continue
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Starting: $exp_id   config: $cfg"
    echo "────────────────────────────────────────────────────────────"

    if python train.py --config "$cfg" $GPU_ARG; then
        echo "  [done] $exp_id"
    else
        echo "  [FAILED] $exp_id — continuing to next"
        FAILED+=("$exp_id")
    fi
done

# ── Report ────────────────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────"
echo "  Generating comparative report..."
echo "────────────────────────────────────────────────────────────"

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    python scripts/orchestrator.py --sweep "$SWEEP_FILE"
else
    echo "  [warn] ANTHROPIC_API_KEY not set — skipping multi-agent report."
    echo "  Set it and run: python scripts/orchestrator.py --sweep $SWEEP_FILE"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Sweep complete                                          ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Monitor:  tensorboard --logdir ./exp/                   ║"
echo "║  Analyze:  python scripts/orchestrator.py                ║"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf "║  FAILED:   %-46s ║\n" "${FAILED[*]}"
fi
echo "╚══════════════════════════════════════════════════════════╝"
