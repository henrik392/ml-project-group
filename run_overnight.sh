#!/bin/bash
# Overnight Experiment Runner
# Runs multiple experiments sequentially with logging

# set -e  # Exit on error (remove this if you want to continue on failures)

# Configuration
LOG_DIR="logs/overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run an experiment with logging
run_experiment() {
    local config=$1
    local name=$(basename "$config" .yaml)
    local log_file="$LOG_DIR/${name}.log"

    echo -e "${BLUE}[$(date +%H:%M:%S)] Starting: $name${NC}"
    echo "Log: $log_file"

    start_time=$(date +%s)

    if uv run python -m src.run_experiment --config "$config" > "$log_file" 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${GREEN}[$(date +%H:%M:%S)] ✓ Completed: $name (${duration}s)${NC}"
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${RED}[$(date +%H:%M:%S)] ✗ Failed: $name (${duration}s)${NC}"
        echo "Check log: $log_file"
        # Uncomment to continue on failure:
        return 0
        # return 1
    fi
}

# Main execution
echo "=========================================="
echo "Overnight Experiment Runner"
echo "Started: $(date)"
echo "Logs: $LOG_DIR"
echo "=========================================="
echo ""

overall_start=$(date +%s)

# List of experiments to run (edit this list as needed)
# Uncomment the set you want to run:

# TEST MODE - Quick 1-epoch validation (use this first!)
# EXPERIMENTS=(
#     "configs/experiments/test/test_exp01.yaml"
#     "configs/experiments/test/test_exp02.yaml"
#     "configs/experiments/test/test_exp05.yaml"
# )

# EXTENDED TRAINING - Train yolo11n to convergence (30 epochs)
EXPERIMENTS=(
    "configs/experiments/exp10_extended_training.yaml"
)

# Add more experiments here as needed:
# "configs/experiments/exp03_conf_sweep.yaml"
# "configs/experiments/exp04_iou_sweep.yaml"
# "configs/experiments/exp06_sahi_inference.yaml"
# "configs/experiments/exp07_bytetrack.yaml"
# "configs/experiments/exp08_sahi_bytetrack.yaml"

# Run all experiments
for experiment in "${EXPERIMENTS[@]}"; do
    run_experiment "$experiment"
    echo ""
done

# Summary
overall_end=$(date +%s)
overall_duration=$((overall_end - overall_start))
hours=$((overall_duration / 3600))
minutes=$(((overall_duration % 3600) / 60))
seconds=$((overall_duration % 60))

echo "=========================================="
echo "All experiments completed!"
echo "Finished: $(date)"
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo "Results: outputs/metrics/results.csv"
echo "Logs: $LOG_DIR"
echo "=========================================="
