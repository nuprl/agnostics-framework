#!/usr/bin/env bash

# Runs a Python command in the venv.
# Can be used for sudo.

script_dir=$(dirname "$(realpath "$0")")
source "$script_dir/activate"

export REPO_ROOT=$(realpath "$script_dir/..")
python -m agnostics.cli.test_executors "$@"
