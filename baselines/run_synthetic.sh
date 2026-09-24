#!/bin/bash
# Run the synthetic benchmark evaluation across all baselines and scenarios
set -e
export PYTHONUNBUFFERED=1

echo "Starting Synthetic Benchmark Baseline Evaluation"

# Scenarios to run: S0 (no interaction), S1 (age only), S2 (age-temporal), S3 (non-linear age)
for scenario in S0 S1 S2 S3; do
    echo "============================================================"
    echo "Running Scenario $scenario"
    echo "============================================================"
    conda run -n ehr python -m baselines.synthetic.runner --scenario "$scenario" --models all

    echo "Running Counterfactual Evaluation for Scenario $scenario"
    conda run -n ehr python -m baselines.synthetic.counterfactual_eval --scenario "$scenario"
done

echo "Evaluation Complete."
