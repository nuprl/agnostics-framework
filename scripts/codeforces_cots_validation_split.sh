#!/usr/bin/env bash

# Runs a Python command in the venv.
# Can be used for sudo.

script_dir=$(dirname "$(realpath "$0")")
source "$script_dir/activate"

python -m agnostics.cli.codeforces_cots.validation_split "$@"
