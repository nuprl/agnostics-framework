#!/usr/bin/env bash

script_dir=$(dirname "$(realpath "$0")")
source "$script_dir/activate"

python -m agnostics.cli.experiments.grpo_mbpp "$@"

# example usage:
# bash scripts/mbpp.sh prepare $lang

# bash scripts/mbpp.sh train --base-model-ref <model_ref_or_path> \
#     --train-ds mbpp-$lang \
#     --test-langs $lang \
#     --vllm-gpu-memory-utilization 0.8 \
#     --no-partial-reward