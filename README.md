# Agnostics
Repository of code developed during the [Agnostics project](https://agnostics.abgru.me).

# Quick start guide
We recommend using `uv`.

```bash
# Create virtual environment
uv venv --python 3.12

# Install dependencies
uv sync
```

The code uses `podman` as the default container tool,
we explain how to configure this in a following "Configuration" section.

We recommend always using scripts in `./scripts` to run code from the project,
as they automatically activate the venv and set recommended environment variables.
You can run `python` with `./scripts/python`.

# Ag-Codeforces-X, Ag-MBPP-X
PL-specialized subsets of both Ag-Codeforces-X and Ag-MBPP-X are generated using code in this repository.

To generate Ag-Codefoces-X splits for programming language $lang (e.g., `lang=lua`), run:
```bash
lang=lua # example
./scripts/grpo.sh prepare $lang
./scripts/grpo.sh prepare xl-varprompt-$lang
```
The relevant generated files are
`./out/grpo/datasets/xl-varprompt-$lang/train.jsonl` (the train split)
and `./out/grpo/datasets/$lang/test.jsonl` (the test split).

To generate Ag-MBPP-X splits, run:
```bash
lang=lua # example
./scripts/grpo_mbpp.sh prepare $lang
```

The identifiers of programming languages used in our work are `lua`, `julia`, `r`, `ocaml`, `fortran`.
A following section explains how to add your own language.

# GRPO training
To train Qwen 3 4B to code in programming language `$lang`, run:
```bash
lang=lua # example
./scripts/grpo.sh train \
    --base-model-ref Qwen/Qwen3-4B \
    --chat-template-type qwen \
    --train-ds ${lang} \
    --partial-reward \
    --group-size 32 \
    --batch-size 4 \
    --micro-batch-size 1 \
    --test-langs ${lang} \
    --custom-test-freq 100 \
    --use-prebuilt-executor-images
```
Base model ref is any model ID accepted by `vllm`.
Chat template type is used to disable thinking in the model
and should be changed for models such as SmolLM3.

To see the available options, run `./scripts/grpo.sh train --help`.

When using a custom programming language, the last flag (`--use-prebuilt-executor-images`) must not be set.
It controls using a prebuilt executor image available in our repository.

# Analyzing trained models
To analyze a trained model on Ag-LiveCodeBench-X, run:
```bash
# The snippet uses Bash features.
lang=lua # example
# model_ref is any model ID supported by vllm, here an example path to a persisted trained model
model_ref=out/grpo/20260220T144110-grpo-multipl/checkpoint_final
# model_nickname is used to name the output directory
model_nickname=my_lua_model
shared=(
    --lang $lang
    --model-nickname "$model_nickname"
    --model-ref "$model_ref"
    --temperature 0.2
    --n-samples 20
    --max-tokens $((1*1024))
    --output-root-dir out
)

./scripts/analysis_model_livecodebenchx.sh "${shared[@]}" generate --batch-size 10000 # high --batch-size is recommended, it controls how many requests vllm receives at once,
./scripts/analysis_model_livecodebenchx.sh "${shared[@]}" verify-from-generate --use-prebuilt-executor-images --timeout-seconds=60
```
The pass@1 scores can be displayed with:
```bash
# to render results in a table, add pipe: | column -t
./scripts/agnostics_pass1_tsv.py ./out/analysis-livecodebenchx/*/*/verify/result.jsonl
```

# Adding a new programming language
You can add support for a custom programming language by adding a short YAML configuration file to `./pl-configs`.
The directory already holds example configurations for the programming languages used in our work.

To generate the necessary code for a programming language `$lang`, run:
```bash
# example: lang=lua
./scripts/gen_pl_code.sh ./pl-configs/$lang.yaml
printf "\n_register_proglang('%s')\n" $lang >> ./src/agnostics/cli/codeforces_cots/proglangs.py
./executors/$lang/build.sh
```
The first step creates a Python module in `./src/agnostics/cli/codeforces_cots/$lang.py`,
an executor image definition in `./executors/$lang`.
The second step registers the language definition in the codebase.
The last step locally builds the executor image for the language.

# Configuration
## The container tool
The container tool used by default is `podman` and can be configured via an environment variable.

You can copy `dotenv-template` to `.env` and adjust the contents, or set the environment variable manually.

## wandb
The training scripts can log to a wandb instance.
All you need to do is log in:

```bash
. ./scripts/activate
wandb login
```

# Executor images
The Agnostics framework tests model-generated code in containers.
The protocol used to communicate with them is documented in `./executors/README.md`.
