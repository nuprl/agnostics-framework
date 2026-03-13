from datetime import datetime
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Iterator, cast
import os
import itertools

from loguru import logger
import ray
import typer
from typing_extensions import Annotated
from prl_ml.grpo.types import Conversation, ChatMessage, BunchedRewardFunction, ModelGenerationData
import datasets

from dockerinator import AgnosticsItem, AgnosticsTestCase, ExecutionResultRow
from dockerinator.agnostics_ray_shim import CodeExecutionActor
from agnostics.util.code_finder import find_final_answer_block
from .codeforces_cots import train_split, validation_split, preprocess_solutions_py, python, lua, julia, c, cpp, r, java, ocaml, fortran
from .grpo import AgnosticsBunchedRewardFunction, make_spawn_code_execution_task
from . import cmd

app = typer.Typer()

OCAML_PROMPT = """
OCaml 5. Some tips:

Numbers:   + - * / mod   vs.   +. -. *. /. **    (add dots!)
Casts:     float_of_int   int_of_float   int_of_string
Mutation:  refs (:= !) or pass new values recursively
Strings:   split_on_char, String.get ⇒ char, use Printf "%c"
Lists:     avoid List.nth; prefer pattern-match / folds / arrays
""".strip()

FORTRAN_PROMPT = """
Fortran 90.

Some tips:
Always begin each scope with implicit none, pick explicit kinds via selected_*_kind, and declare proper lengths—character(len=*) is legal only for dummy arguments, not locals.
Strings are blank-padded: call len_trim before iterating, and store dynamic text in deferred-length allocatables (character(len=:), allocatable :: s).
List-directed read(*,*) arr does not auto-size arrays; read a count first, then allocate and read, or tokenize a line manually.
When translating 0-based formulas (heaps, bit positions) remember Fortran arrays default to 1-based; if you want 0-based, declare lower bounds.
Use real literals (2.0d0, 1.0_rk) to avoid silent integer division, and guard against overflow when exponentiating integers.
For frequency tables, allocate an array or use findloc; Fortran lacks native dicts/sets, so you must implement search yourself.
Prefer array intrinsics (sum, count, pack) over hand-rolled loops, and keep helper procedures inside a contains section or module so interfaces are explicit.
return inside the main program is non-idiomatic; use structured blocks or stop.
Never print interactive prompts in batch solutions; just read, compute, and write.
""".strip()

LANGS = ('python', 'cpp', 'lua', 'java', 'r', 'julia', 'ocaml', 'fortran')
LANG_STR_LONG = {
    'python': 'Python 3',
    'cpp': 'C++ 17',
    'lua': 'Lua 5.1 which targets LuaJIT',
    'java': 'Java 24',
    'ocaml': OCAML_PROMPT,
    'r': 'R version 4',
    'julia': 'Julia 1.11',
    'fortran': FORTRAN_PROMPT,
}
DATASETS = ["mbpp-python", "mbpp-cpp", "mbpp-lua", "mbpp-java", "mbpp-r", "mbpp-julia", "mbpp-ocaml", "mbpp-fortran"]

PROMPT_TEMPLATE = """
Your task is to solve a programming problem.

# Problem

{problem_statement}

# Input Format

{input_format}

# Output Format

{output_format}

# Examples

{examples}

# Instructions

Analyze the maximum input constraints and identify the optimal algorithmic approach and data structures needed to process the largest possible test cases within the time and memory limits, then explain why your chosen implementation strategy is the most efficient solution.

Please reason step by step about your solution approach, then provide a complete implementation in {lang_str_long}.

Put your final answer in a single code block, for example:
```{lang_str}
<your code here>
```
""".strip()


def build_chat_message_prompt(
    item: dict[str, Any],
    completion: str | None
) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    msgs.append({ 'role': 'system', 'content': '/nothink' })
    msgs.append({ 'role': 'user', 'content': item['prompt'] })
    if completion is not None:
        msgs.append({ 'role': 'assistant', 'content': completion })
    return msgs


@app.command()
def prepare(
    lang: Annotated[str, typer.Argument()],
):
    ctx = cmd.start(__file__, sub_out_d=f'{lang}', force=True)

    assert lang in LANGS

    ds_gen = datasets.load_dataset("nuprl/mbpp-agnostic-translation", "full", split='train')

    original_test_ids = range(11, 510+1)
    original_fewshot_ids = range(1, 11)
    original_train_ids = range(601, 974+1)

    train_ds = ds_gen.filter(lambda x: x["original_task_id"] in original_train_ids or x["original_task_id"] in original_fewshot_ids)
    test_ds = ds_gen.filter(lambda x: x["original_task_id"] in original_test_ids)

    train_ds = train_ds.map(lambda x: {
        "idx": x["original_task_id"],
        "prompt": PROMPT_TEMPLATE.format(
            problem_statement=x["description"],
            examples="## Input\n" + x["tests"][0]["input"] + "\n\n## Output\n" + x["tests"][0]["output"],
            lang_str_long=LANG_STR_LONG[lang],
            lang_str=lang,
            input_format=x["input_format"],
            output_format=x["output_format"],
        ),
        "test_cases": x["tests"],
        "lang": lang,
    })

    test_ds = test_ds.map(lambda x: {
        "idx": x["original_task_id"],
        "prompt": PROMPT_TEMPLATE.format(
            problem_statement=x["description"],
            examples="## Input\n" + x["tests"][0]["input"] + "\n\n## Output\n" + x["tests"][0]["output"],
            lang_str_long=LANG_STR_LONG[lang],
            lang_str=lang,
            input_format=x["input_format"],
            output_format=x["output_format"],
        ),
        "test_cases": x["tests"],
        "lang": lang,
    })

    train_ds.to_json(ctx.out_d/'train.jsonl', orient='records', lines=True)
    test_ds.to_json(ctx.out_d/'test.jsonl', orient='records', lines=True)

    logger.success('Wrote: {}', cmd.cwd_rel(ctx.out_d/'train.jsonl'))
    logger.success('Wrote: {}', cmd.cwd_rel(ctx.out_d/'test.jsonl'))



@app.command()
def train(
    base_model_ref: Annotated[str, typer.Option()],
    timestamp: str | None = None,
    train_ds: str | None = None,
    epochs: int = 1,
    temperature: float | None = None,
    scheduler_spec: str | None = None,
    group_size: int | None = None,
    partial_reward: bool = False,
    test_langs: Annotated[str | None, typer.Option(
        help="""
        A comma-separated list of languages to test on.
        """
    )] = None,
    custom_test_freq: int | None = None,
    train_data_shuffle: bool = True,
    batch_size: int | None = None,
    micro_batch_size: int | None = None,
    vllm_gpu_memory_utilization: Annotated[float | None, typer.Option(
        help="""
        See vLLM documentation. Larger models need lower values.
        Recommmended: 0.85 for 4B and 0.90 for <2B.
        """
    )] = None,
):
    assert epochs > 0

    split_test_langs = (test_langs or '').split(',')
    print(split_test_langs, test_langs)
    if not (all(lang in LANGS for lang in split_test_langs)):
        raise typer.BadParameter(f'--test-langs must be a comma-separated subset of: {", ".join(LANGS)}')
    if test_langs is None:
        resolved_test_langs = None
    else:
        resolved_test_langs = split_test_langs

    if train_ds is not None and train_ds not in DATASETS:
        raise typer.BadParameter(f'--train-ds must be one of: {", ".join(DATASETS)}')
    run_name = 'grpo-mbpp'

    if not timestamp:
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    grpo_run_tags = []
    if epochs > 1:
        grpo_run_tags.append(f'epochs{epochs}')
    if group_size:
        grpo_run_tags.append(f'groupsz{group_size}')
    if batch_size:
        grpo_run_tags.append(f'batchsz{batch_size}')
    if micro_batch_size:
        grpo_run_tags.append(f'microbatchsz{micro_batch_size}')
    if grpo_run_tags:
        grpo_run_tags.insert(0, '_')
    grpo_run_name = f'{run_name}-{train_ds.replace('_', '-')}{"_".join(grpo_run_tags)}@{timestamp}'

    if sjobid := os.getenv('SLURM_JOB_ID'):
        logger.info('SLURM_JOB_ID: {}', sjobid)
        grpo_run_name += f'_j{sjobid}'

    do_train(
        timestamp=timestamp,
        base_model_ref=base_model_ref,
        train_ds=train_ds,
        epochs=epochs,
        temperature=temperature,
        group_size=group_size,
        partial_reward=partial_reward,
        scheduler_str=scheduler_spec,
        test_langs=resolved_test_langs,
        train_data_shuffle=train_data_shuffle,
        custom_test_freq=custom_test_freq,
        run_name=run_name,
        grpo_run_name=grpo_run_name,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    )


def do_train(
    timestamp: str | None,
    base_model_ref: str,
    train_ds: str | None = None,
    epochs: int = 1,
    temperature: float | None = None,
    scheduler_str: str | None = None,
    group_size: int | None = None,
    partial_reward: bool = False,
    test_langs: list[str] | None = None,
    custom_test_freq: int | None = None,
    train_data_shuffle: bool = True,
    batch_size: int | None = None,
    micro_batch_size: int | None = None,
    vllm_gpu_memory_utilization: float | None = None,
    containers_per_pl: int | None = None,
    run_name: str | None = None,
    out_d: Path | None = None,
    grpo_run_name: str | None = None,
):
    import prl_ml.grpo.ray_trainer

    timestamp = timestamp or datetime.now().strftime('%Y%m%dT%H%M%S')
    train_ds = train_ds or 'mbpp-lua'
    run_name = run_name or 'run'

    temperature = temperature or 0.7
    scheduler_str = scheduler_str or 'cosine(warmup_ratio=0.1)'
    group_size = group_size or 32
    batch_size = batch_size or 4
    micro_batch_size = micro_batch_size or 1
    vllm_gpu_memory_utilization = vllm_gpu_memory_utilization or 0.90
    containers_per_pl = containers_per_pl or group_size

    assert epochs > 0
    if test_langs is None:
        test_langs = ('python', 'cpp', 'lua')
    assert len(test_langs) == len(set(test_langs))
    assert all(lang in LANGS for lang in test_langs)
    assert train_ds in DATASETS

    os.environ['VLLM_USE_V1'] = '0'

    ctx = cmd.start(__file__, out_d=out_d, sub_out_d=f'{timestamp}-{run_name}')

    root_datasets_d = ctx.out_d.parent
    print("root_datasets_d", root_datasets_d)
    out_io_d = ctx.out_d/'container-io-logs'
    out_io_d.mkdir(parents=True, exist_ok=True)

    train_langs = train_ds.split('-')[1:]
    print("train_langs", train_langs)
    print("test_langs", test_langs)
    assert train_langs
    container_langs = train_langs
    container_langs.extend(set(test_langs) - set(train_langs))

    print("container_langs", container_langs)
    actors = {}
    for lang in container_langs:
        lang_out_io_d = out_io_d/lang
        lang_out_io_d.mkdir(parents=True, exist_ok=True)
        actors[lang] = CodeExecutionActor.remote(
            executor_image_name=f'agnostics-{lang}-executor',
            worker_num=containers_per_pl,
            io_log_dir=lang_out_io_d,
        )
    if len(train_langs) > 1:
        def spawn_code_execution_task(item_id: str, item: AgnosticsItem) -> ray.ObjectRef:
            a = actors[item.lang]
            return a.run_code.remote(item_id, item)
        reward_funcs = AgnosticsBunchedRewardFunction(
            spawn_code_execution_task,
            partial_reward,
        )
    else:
        assert len(train_langs) == 1
        reward_funcs = AgnosticsBunchedRewardFunction(
            make_spawn_code_execution_task(actors[train_langs[0]]),
            partial_reward,
        )

    checkpoint_freq = 0

    train_item_len_limit = 3000

    prl_ml.grpo.ray_trainer.train(
        model_name=base_model_ref,
        optimizer_str='AdamW(lr=5e-6 betas=(0.9, 0.999))',
        scheduler_str=scheduler_str,
        run_dir=str(ctx.out_d),
        reward_funcs=reward_funcs,
        test_rewards_dict={
            lang: AgnosticsBunchedRewardFunction(
                make_spawn_code_execution_task(actor),
                partial_reward,
            )
            for lang, actor in actors.items()
        },
        custom_test_freq=custom_test_freq,
        prompt_builder=build_chat_message_prompt,
        batch_size=batch_size,
        group_size=group_size,
        micro_batch_size=micro_batch_size,
        num_epochs=epochs,
        stop=[],
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        beta=0.0,
        concurrent_group_generation_hack=True,
        group_max_tokens=1024,
        clipping_epsilon=0.1,
        group_sample_temperature=temperature,
        group_output_file=ctx.out_d/'group_output.jsonl',
        train_dataset_spec=f'jsonl:{root_datasets_d}/{lang}/train.jsonl',
        test_dataset_spec={
            lang: f'jsonl:{root_datasets_d}/{lang}/test.jsonl'
            for lang in test_langs
        },
        train_item_len_limit=train_item_len_limit,
        train_data_shuffle=train_data_shuffle,
        checkpoint_freq=checkpoint_freq,
        project_name=f'agnostics-grpo-{train_ds}',
        run_name=grpo_run_name,
    )


if __name__ == '__main__':
    app()
