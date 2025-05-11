from datetime import datetime
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Iterator, Literal, cast
import os
import itertools

from agnostics.cli.codeforces_cots import proglangs
from loguru import logger
import ray
import typer
from typing_extensions import Annotated
from prl_ml.grpo.types import ChatMessage, BunchedRewardFunction, ModelGenerationData

from dockerinator import AgnosticsItem, AgnosticsTestCase, ExecutionResultRow
from dockerinator.agnostics_ray_shim import CodeExecutionActor
from agnostics.util.code_finder import find_final_answer_block
from .codeforces_cots import (
    train_split, test_split, preprocess_solutions_py,
)
from . import cmd

app = typer.Typer()


type SpawnCodeExecutionTaskFn = Callable[[str, AgnosticsItem], ray.ObjectRef]


class AgnosticsBunchedRewardFunction(BunchedRewardFunction):
    def __init__(
        self,
        spawn_code_execution_task_fn,
        partial_reward: bool,
    ):
        self._spawn_code_execution_task_fn = spawn_code_execution_task_fn
        self._partial_reward = partial_reward

    def compute_reward(self, generation_data: list[ModelGenerationData]) -> list[tuple[str, list[float]]]:

        tasksOrRewards = []
        for d in generation_data:
            idx = d.row['idx']
            test_cases = [
                AgnosticsTestCase(input=example['input'], output=example['output'])
                for example in d.row['test_cases']
            ]
            lang = d.row.get('lang', None)
            code = find_final_answer_block(d.output, offset=0, answer_must_be_valid_python=False)
            if code is None:
                tasksOrRewards.append(0.0)
            else:
                tasksOrRewards.append(self._spawn_code_execution_task_fn(
                    str(idx), AgnosticsItem(code=code, test_cases=test_cases, lang=lang)
                ))

        responsesOrRewards: list[Any] = [
            elt if isinstance(elt, float) else ray.get(elt)
            for elt in tasksOrRewards
        ]

        proper_rewards = []
        partial_rewards = []
        for elt in responsesOrRewards:
            if isinstance(elt, float):
                partial_rewards.append(0.0)
                proper_rewards.append(elt)
            else:
                elt = cast(ExecutionResultRow, elt)
                proper_rewards.append(
                    1.0 if elt.status == 'success' else 0.0
                )
                partial_rewards.append(
                    0.2 if elt.status == 'fail:wrong-output' else 0.0
                )
        res = [('proper', proper_rewards)]
        if self._partial_reward:
            res.append(('partial', partial_rewards))
        return res


def make_spawn_code_execution_task(
    actor: Any,
):
    def spawn_code_execution_task(item_id: str, item: AgnosticsItem) -> ray.ObjectRef:
        return actor.run_code.remote(item_id, item)
    return spawn_code_execution_task


type BuildPromptFn = Callable[[preprocess_solutions_py.SolutionsRow], str]

LANGS = tuple(proglangs.PROGLANGS)
LANG_MIXES = ('mix-cpp-lua', 'mix-python-lua', 'mix-lua-julia', 'mix-python-cpp-lua')
SPECIAL_DATASETS = (
    *[f'xl-varprompt-{lang}' for lang in LANGS],
    'xl-varprompt-mix-lua-julia-r-ocaml-fortran',
    'xl-varprompt-mix-lua-julia-r',
    'xl-varprompt-fullmix-lua-julia-r-ocaml-fortran',
    'xl-varprompt-fullmix-lua-julia-r',
    'seq-python-lua__preshuffled_epochs2',
    'fullmix-python-cpp-lua__epochs3',
    'fullmix-python-cpp-lua-julia-r-java__epochs6',
)
DATASETS = (*LANGS, *LANG_MIXES, *SPECIAL_DATASETS)

_lang_mix_prefixes = ('mix-', 'fullmix-', 'seq-', 'xl-varprompt-mix-', 'xl-varprompt-fullmix-')

def ds_is_lang_mix(ds: str) -> bool:
    return ds.startswith(_lang_mix_prefixes)

def ds_is_xl(ds: str) -> bool:
    return ds.startswith('xl-')

def ds_is_varprompt(ds: str) -> bool:
    return ds.startswith('xl-varprompt-')

def ds_get_langs(ds: str) -> list[str]:
    if ds_is_lang_mix(ds):
        for p in _lang_mix_prefixes:
            ds = ds.removeprefix(p)
        return ds.split('_', 1)[0].split('-')
    elif ds_is_xl(ds):
        return [ds.removeprefix('xl-').removeprefix('varprompt-')]
    else:
        return [ds]

def ds_extra_epochs_len(ds: str) -> int | None:
    split0 = ds.split('__')
    if len(split0) == 1:
        return None
    assert len(split0) == 2
    split = split0[1].split('_')
    relevant_tag = 'epochs'
    for s in split:
        if s.startswith(relevant_tag):
            return int(s.removeprefix(relevant_tag))
    return None

def ds_is_preshuffled(ds: str) -> bool:
    split0 = ds.split('__')
    if len(split0) == 1:
        return False
    assert len(split0) == 2
    split = split0[1].split('_')
    return 'preshuffled' in split


def _gen_output_rows(
    rng: random.Random,
    langs: list[str],
    in_rows: Iterable[preprocess_solutions_py.SolutionsRow],
    use_variant_prompt: bool = False,
) -> Iterator[dict[str, Any]]:
    assert langs
    for in_r in in_rows:
        lang = rng.choice(langs)
        build_prompt_cmd = proglangs.BUILD_PROMPT_CMDS[lang]
        def _build_prompt(in_r: preprocess_solutions_py.SolutionsRow) -> str:
            if use_variant_prompt:
                return build_prompt_cmd.build_variant_prompt(rng, in_r)
            else:
                return build_prompt_cmd.build_prompt(in_r, allow_omitting_only_example=True)
        r = {
            'idx': in_r.idx,
            'test_cases': [
                { 'input': example.input, 'output': example.output }
                for example in in_r.examples
            ],
            'lang': lang,
            'prompt': _build_prompt(in_r),
        }
        yield r


def _write_datasets(
    out_d: Path,
    langs: list[str],
):
    out_d.mkdir(parents=True, exist_ok=True)
    out_train_f = out_d/'train.jsonl'
    out_test_f = out_d/'test.jsonl'

    cmd.ser.jsonl_dumpf(
        _gen_output_rows(random.Random(42), langs, train_split.gen_from_hf()),
        out_train_f,
    )
    logger.success('Wrote: {}', cmd.cwd_rel(out_train_f))

    cmd.ser.jsonl_dumpf(
        _gen_output_rows(random.Random(42), langs, test_split.gen_from_hf()),
        out_test_f,
    )
    logger.success('Wrote: {}', cmd.cwd_rel(out_test_f))


def _write_xl_datasets(
    out_d: Path,
    langs: list[str],
    use_variant_prompt: bool,
):
    out_d.mkdir(parents=True, exist_ok=True)
    out_train_f = out_d/'train.jsonl'

    cmd.ser.jsonl_dumpf(
        _gen_output_rows(
            random.Random(42),
            langs,
            train_split.gen_xl_from_hf(),
            use_variant_prompt=use_variant_prompt,
        ),
        out_train_f,
    )
    logger.success('Wrote: {}', cmd.cwd_rel(out_train_f))


def _gen_preshuffled_from_hf(split: str, make_solutions_row_fn):
    import datasets
    from tqdm import tqdm
    ds = datasets.load_dataset('nuprl/agnostics-codeforces-cots', split=split)
    ds = ds.shuffle(seed=42)

    for in_row in tqdm(ds, desc=f'Processing the {split} split'):
        yield make_solutions_row_fn(in_row)

def _gen_preshuffled_train_from_hf():
    return _gen_preshuffled_from_hf('train', train_split.make_solutions_row)


def _write_seq_python_lua__epochs2(root_out_d: Path):
    out_d = root_out_d/'seq-python-lua__preshuffled_epochs2'
    assert out_d.name in SPECIAL_DATASETS
    out_d.mkdir(parents=True, exist_ok=True)

    out_train_f = out_d/'train.jsonl'

    def _chain(iters):
        for iter in iters:
            for elem in iter:
                yield elem

    cmd.ser.jsonl_dumpf(
        _chain([
            _gen_output_rows(random.Random(42), ['python'], _gen_preshuffled_train_from_hf()),
            _gen_output_rows(random.Random(42), ['lua'], _gen_preshuffled_train_from_hf()),
        ]),
        out_train_f,
    )
    logger.success('Wrote: {}', cmd.cwd_rel(out_train_f))


def _write_fullmix(root_out_d: Path, ds_name: str):
    assert ds_name.startswith('fullmix-')
    out_d = root_out_d/ds_name
    out_d.mkdir(parents=True, exist_ok=True)

    out_train_f = out_d/'train.jsonl'

    langs = ds_get_langs(ds_name)

    def _chain(iters):
        for iter in iters:
            for elem in iter:
                yield elem

    cmd.ser.jsonl_dumpf(
        _chain([
            _gen_output_rows(random.Random(42), [l], train_split.gen_from_hf())
            for l in langs
        ]),
        out_train_f,
    )
    logger.success('Wrote: {}', cmd.cwd_rel(out_train_f))


def _write_xl_fullmix(root_out_d: Path, ds_name: str, use_variant_prompt: bool):
    assert ds_name.startswith('xl-varprompt-fullmix-')
    out_d = root_out_d/ds_name
    out_d.mkdir(parents=True, exist_ok=True)

    out_train_f = out_d/'train.jsonl'

    langs = ds_get_langs(ds_name)

    def _chain(iters):
        for iter in iters:
            for elem in iter:
                yield elem

    cmd.ser.jsonl_dumpf(
        _chain([
            _gen_output_rows(
                random.Random(42),
                [l],
                train_split.gen_xl_from_hf(),
                use_variant_prompt=use_variant_prompt,
            )
            for l in langs
        ]),
        out_train_f,
    )
    logger.success('Wrote: {}', cmd.cwd_rel(out_train_f))


def build_qwen_chat_message_prompt(
    item: dict[str, Any],
    completion: str | None
) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    msgs.append({ 'role': 'system', 'content': '/nothink' })
    msgs.append({ 'role': 'user', 'content': item['prompt'] })
    if completion is not None:
        msgs.append({ 'role': 'assistant', 'content': completion })
    return msgs


def build_smollm3_chat_message_prompt(
    item: dict[str, Any],
    completion: str | None
) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    msgs.append({ 'role': 'system', 'content': '/no_think' })
    msgs.append({ 'role': 'user', 'content': item['prompt'] })
    if completion is not None:
        msgs.append({ 'role': 'assistant', 'content': completion })
    return msgs


@app.command()
def prepare(
    ds: Annotated[str | None, typer.Argument()] = None,
):
    def _output_pure_ds(root_out_d: Path, lang: str):
        out_d = root_out_d/lang
        _write_datasets(out_d, [lang])

    def _output_mix_ds(root_out_d: Path, ds_name: str):
        out_d = root_out_d/ds_name
        _write_datasets(out_d, ds_get_langs(ds_name))

    def _output_special_ds(root_out_d: Path, ds_name: str):
        if ds_name == 'seq-python-lua__preshuffled_epochs2':
            _write_seq_python_lua__epochs2(root_out_d=root_out_d)
        elif ds_name.startswith('fullmix-'):
            _write_fullmix(root_out_d=root_out_d, ds_name=ds_name)
        elif ds_name.startswith('xl-varprompt-fullmix-'):
            _write_xl_fullmix(root_out_d=root_out_d, ds_name=ds_name, use_variant_prompt=True)
        elif ds_name.startswith('xl-'):
            _write_xl_datasets(
                out_d=root_out_d/ds_name,
                langs=ds_get_langs(ds_name),
                use_variant_prompt=ds_is_varprompt(ds_name),
            )
        else:
            raise ValueError(f'expected a special dataset, got: {ds_name=}')

    ctx = cmd.start(
        __file__,
        sub_out_d='datasets',
        force=True,
    )

    if ds is None:
        for lang in proglangs.BUILD_PROMPT_CMDS.keys():
            _output_pure_ds(ctx.out_d, lang)

        for ds in LANG_MIXES:
            _output_mix_ds(ctx.out_d, ds)

        for ds in SPECIAL_DATASETS:
            _output_special_ds(ctx.out_d, ds)
    else:
        if ds in proglangs.BUILD_PROMPT_CMDS:
            _output_pure_ds(ctx.out_d, ds)
        elif ds in LANG_MIXES:
            _output_mix_ds(ctx.out_d, ds)
        elif ds in SPECIAL_DATASETS:
            _output_special_ds(ctx.out_d, ds)
        else:
            logger.warning('known datasets: \n{}', '\n'.join(DATASETS))
            raise typer.BadParameter(f'unrecognized dataset: {ds}')


@app.command()
def train(
    base_model_ref: Annotated[str, typer.Option()],
    chat_template_type: str = 'qwen',
    timestamp: str | None = None,
    train_ds: str | None = None,
    epochs: int = 1,
    temperature: float | None = None,
    learning_rate: str | None = None,
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
    containers_per_pl: int | None = None,
    use_prebuilt_executor_images: bool = False,
    custom_checkpoint_freq: int | None = None,
):
    assert epochs > 0

    split_test_langs = [l for l in (test_langs or '').split(',') if l]
    print(split_test_langs, test_langs)
    if not (all(lang in LANGS for lang in split_test_langs)):
        raise typer.BadParameter(f'--test-langs must be a comma-separated subset of: {", ".join(LANGS)}')
    if test_langs is None:
        resolved_test_langs = None
    else:
        resolved_test_langs = split_test_langs

    try:
        if learning_rate:
            float(learning_rate)
    except ValueError:
        raise typer.BadParameter(f'--learning-rate must be a float, got: {learning_rate}')

    if train_ds is not None and train_ds not in DATASETS:
        raise typer.BadParameter(f'--train-ds must be one of: {", ".join(DATASETS)}')

    run_name = 'grpo-multipl'

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
        chat_template_type=chat_template_type,
        train_ds=train_ds,
        epochs=epochs,
        temperature=temperature,
        group_size=group_size,
        partial_reward=partial_reward,
        learning_rate_str=learning_rate,
        scheduler_str=scheduler_spec,
        test_langs=resolved_test_langs,
        train_data_shuffle=train_data_shuffle,
        custom_test_freq=custom_test_freq,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        containers_per_pl=containers_per_pl,
        use_prebuilt_executor_images=use_prebuilt_executor_images,
        custom_checkpoint_freq=custom_checkpoint_freq,
        run_name=run_name,
        grpo_run_name=grpo_run_name,
    )


def do_train(
    timestamp: str | None,
    base_model_ref: str,
    chat_template_type: Literal['qwen', 'smollm3'],
    train_ds: str | None = None,
    epochs: int = 1,
    temperature: float | None = None,
    learning_rate_str: str | None = None,
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
    use_prebuilt_executor_images: bool = False,
    custom_checkpoint_freq: int | None = None,
    run_name: str | None = None,
    out_d: Path | None = None,
    grpo_run_name: str | None = None,
):
    import prl_ml.grpo.ray_trainer

    timestamp = timestamp or datetime.now().strftime('%Y%m%dT%H%M%S')
    train_ds = train_ds or 'lua'
    run_name = run_name or 'run'

    temperature = temperature or 0.7
    learning_rate_str = learning_rate_str or '5e-6'
    scheduler_str = scheduler_str or 'cosine(warmup_ratio=0.1)'
    group_size = group_size or 32
    batch_size = batch_size or 4
    micro_batch_size = micro_batch_size or 1
    vllm_gpu_memory_utilization = vllm_gpu_memory_utilization or 0.90
    containers_per_pl = containers_per_pl or group_size

    assert epochs > 0
    if test_langs is None:
        test_langs = ['python', 'cpp', 'lua']
    elif test_langs == []:
        raise ValueError('Currently test_langs must be non-empty, see comment.')
    assert len(test_langs) == len(set(test_langs))
    assert all(lang in LANGS for lang in test_langs)
    assert train_ds in DATASETS
    try:
        float(learning_rate_str)
    except ValueError:
        raise AssertionError(f'Not a float: {learning_rate_str=!r}')

    os.environ['VLLM_USE_V1'] = '0'

    ctx = cmd.start(__file__, out_d=out_d, sub_out_d=f'{timestamp}-{run_name}')
    logger.info('Starting training run in dir: {}', ctx.out_d)

    root_datasets_d = ctx.out_d.parent/'datasets'
    out_io_d = ctx.out_d/'container-io-logs'
    out_io_d.mkdir(parents=True, exist_ok=True)

    train_langs = ds_get_langs(train_ds)
    assert train_langs
    container_langs = train_langs
    container_langs.extend(set(test_langs) - set(train_langs))
    logger.warning('container_langs: {!r}', container_langs)

    def get_image_name(lang: str) -> str:
        if use_prebuilt_executor_images:
            return proglangs.PREBUILT_IMAGE_NAMES[lang]
        return proglangs.IMAGE_NAMES[lang]


    actors = {}
    for lang in container_langs:
        lang_out_io_d = out_io_d/lang
        lang_out_io_d.mkdir(parents=True, exist_ok=True)
        actors[lang] = CodeExecutionActor.remote(
            executor_image_name=get_image_name(lang),
            worker_num=containers_per_pl,
            io_log_dir=lang_out_io_d,
        )
    if len(container_langs) > 1:
        def spawn_code_execution_task(item_id: str, item: AgnosticsItem) -> ray.ObjectRef:
            a = actors[item.lang]
            return a.run_code.remote(item_id, item)
        reward_funcs = AgnosticsBunchedRewardFunction(
            spawn_code_execution_task,
            partial_reward,
        )
    else:
        assert len(container_langs) == 1
        reward_funcs = AgnosticsBunchedRewardFunction(
            make_spawn_code_execution_task(actors[container_langs[0]]),
            partial_reward,
        )

    if custom_checkpoint_freq:
        checkpoint_freq = custom_checkpoint_freq
    else:
        checkpoint_freq = 0
        if special_epochs_len := ds_extra_epochs_len(train_ds):
            assert epochs == 1, f'probably a mistake: {epochs=}'
            assert batch_size == 4, f'unexpected: {batch_size=}'
            checkpoint_freq = 500
        elif ds_is_xl(train_ds):
            checkpoint_freq = 500

    if ds_is_preshuffled(train_ds):
        assert train_data_shuffle is False, f'a mistake: {train_data_shuffle=}'

    train_item_len_limit = 3000

    if chat_template_type == 'qwen':
        prompt_builder = build_qwen_chat_message_prompt
    elif chat_template_type == 'smollm3':
        prompt_builder = build_smollm3_chat_message_prompt
    else:
        raise ValueError(f'unknown chat template type: {chat_template_type}')
    logger.info('Chat template type: {}', chat_template_type)
    logger.info('Resolved prompt builder to (name): {}', prompt_builder.__name__)
    if base_model_ref == 'HuggingFaceTB/SmolLM3-3B':
        assert chat_template_type == 'smollm3', f'wrong setting: {chat_template_type=}'

    prl_ml.grpo.ray_trainer.train(
        model_name=base_model_ref,
        optimizer_str=f'AdamW(lr={learning_rate_str} betas=(0.9, 0.999))',
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
        prompt_builder=prompt_builder,
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
        train_dataset_spec=f'jsonl:{root_datasets_d/train_ds}/train.jsonl',
        test_dataset_spec={
            lang: f'jsonl:{root_datasets_d/lang}/test.jsonl'
            for lang in test_langs
        },
        train_item_len_limit=train_item_len_limit,
        train_data_shuffle=train_data_shuffle,
        checkpoint_freq=checkpoint_freq,
        project_name='agnostics-grpo-qwen3-1p7b',
        run_name=grpo_run_name,
    )


if __name__ == '__main__':
    app()
