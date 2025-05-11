#!/usr/bin/env bash

script_dir=$(dirname "$(realpath "$0")")
source "$script_dir/activate"

python -m agnostics.cli.analysis.model_livecodebenchx "$@"
