#!/bin/bash

# Configuration
START_ID=0
END_ID=499
MAX_PARALLEL=${1:-4}  # Default to 4 parallel jobs

# Job parameters
declare models=("noisy_q_learning")

# Create logs directory if it doesn't exist
mkdir -p logs

# Delete .DS_Store files
find . -name ".DS_Store" -delete

echo "Running m5_recovery_fit.py with job IDs $START_ID to $END_ID"
echo "Max parallel jobs: $MAX_PARALLEL"

# Parallel execution using xargs: does not work on mac
for model in "${models[@]}"; do
    seq $START_ID $END_ID | xargs -P $MAX_PARALLEL -I {} bash -c "{ echo 'Starting job {}: Model=$model...'; python m5_recovery_fit.py -n $model -s 420 {} 2>&1 && echo 'Job {} completed successfully' || echo 'Job {} failed'; } > logs/restless_bandit-fit_{}.out 2>&1"
done

echo "All jobs completed!"
