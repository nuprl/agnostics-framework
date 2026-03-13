import asyncio
import base64
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from os import environ, getenv
from pathlib import Path
import pickle
from typing import Annotated, Any
import zlib

import typer
from loguru import logger
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from .. import cmd
from ..codeforces_cots import proglangs
from ..codeforces_cots.common.commands import CmdExtractAnswersInstance

app = typer.Typer()


@dataclass
class AnalysisArgs:
    model_ref: str
    model_config: str | None
    model_nickname: str
    output_root_dir: Path
    lang: str
    temperature: float
    temperature_str: str
    n_samples: int
    max_tokens: int


ANALYSIS_ARGS: AnalysisArgs | None = None


def analysis_wd_from_run_ref(
    output_root_dir: Path,
    model_nickname: str,
    lang: str,
    temperature_str: str,
    n_samples: int,
    max_tokens: int,
) -> Path:
    clean_temperature_str = temperature_str.replace(".", "p")
    return (
        output_root_dir
        / "analysis-livecodebenchx"
        / model_nickname
        / f"{lang}__n{n_samples}_maxtoks{max_tokens}_tmpr{clean_temperature_str}"
    )


def analysis_wd_from_args(args: AnalysisArgs) -> Path:
    return analysis_wd_from_run_ref(
        output_root_dir=args.output_root_dir,
        model_nickname=args.model_nickname,
        lang=args.lang,
        temperature_str=args.temperature_str,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
    )


@app.callback()
def set_shared_analysis_args(
    lang: Annotated[str, typer.Option(help="The PL to generate code for.")],
    temperature: Annotated[str, typer.Option(help="The temperature to use for generation.")],
    n_samples: Annotated[int, typer.Option(help="The number of samples to generate.")],
    max_tokens: Annotated[int, typer.Option(help="The maximum number of tokens to generate.")],
    model_ref: Annotated[str, typer.Option(help=("A ref/name of the model to generate code with."))],
    model_nickname: Annotated[str, typer.Option(help=("The nickname of the model to generate code with."))],
    output_root_dir: Annotated[
        Path,
        typer.Option(
            help=(
                "The root directory under which the analysis output will be stored,"
                " in a subdirectory named after the arguments."
            )
        ),
    ],
    model_config: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional config name kept for CLI compatibility with other analysis entrypoints."
                " (Not used by the API backend.)"
            )
        ),
    ] = None,
) -> None:
    """
    Sets analysis arguments shared between the steps.

    Some of the arguments are shared just to determine the working directory.
    """
    global ANALYSIS_ARGS

    assert not ANALYSIS_ARGS

    if lang not in proglangs.PROGLANGS:
        raise typer.BadParameter(f"--lang must be one of: {', '.join(proglangs.PROGLANGS)}")

    temperature_str = temperature
    try:
        temperature_float = float(temperature_str)
    except ValueError:
        raise typer.BadParameter("--temperature must be a float.")
    assert n_samples > 0

    ANALYSIS_ARGS = AnalysisArgs(
        model_ref=model_ref,
        model_config=model_config,
        model_nickname=model_nickname,
        output_root_dir=output_root_dir,
        lang=lang,
        temperature=temperature_float,
        temperature_str=temperature_str,
        n_samples=n_samples,
        max_tokens=max_tokens,
    )


@app.command()
def generate(
    concurrency: Annotated[int, typer.Option(help="Client-side concurrency (in-flight requests).")],
    force: bool = False,
    send_one_request_per_sample: bool = False,
    api_base_url: Annotated[str | None, typer.Option(
        help="OpenAI-compatible API base URL (e.g. http://localhost:8890/v1)."
    )] = None,
    timeout_seconds: Annotated[float, typer.Option(
        help="Per-request timeout in seconds."
    )] = 10.0,
) -> None:
    import datasets

    def make_prompt_from_lcbx_row(row: dict, lang_prompt: str) -> str:
        return f"""\
# Problem
{row['question_content']}

# Task
Provide a full implementation of the specified program in a Markdown code block.
Use the following programming language: {lang_prompt}
"""

    args = ANALYSIS_ARGS
    assert args is not None
    assert concurrency > 0

    wd = analysis_wd_from_args(args)
    out_d = wd / "generate"
    if not force and out_d.exists():
        logger.error("Output dir already exists: {}", cmd.cwd_rel(out_d))
        logger.error("Use --force to overwrite an existing output dir.")
        raise typer.Exit(1)
    out_d.mkdir(parents=True, exist_ok=True)
    cmd.start_no_ctx(out_d)

    out_config_f = out_d / "config.json"
    out_f = out_d / "result.jsonl"

    cmd.ser.json_dumpf({
        "backend": "openai-compatible",
        "model_ref": args.model_ref,
        "model_config": args.model_config,
        "model_nickname": args.model_nickname,
        "lang": args.lang,
        "temperature": args.temperature,
        "n_samples": args.n_samples,
        "max_tokens": args.max_tokens,
        "batch_size": concurrency,
        "override_api_base_url": api_base_url,
        "timeout_seconds": timeout_seconds,
    }, out_config_f)
    logger.info("Wrote step config: {}", cmd.cwd_rel(out_config_f))

    ds = datasets.load_dataset("nuprl/Ag-LiveCodeBench-X", split="test")
    lang_prompt = args.lang
    rows: list[dict[str, Any]] = [
        {"prompt": make_prompt_from_lcbx_row(row, lang_prompt=lang_prompt), "idx": row["question_id"]}
        for row in ds
    ]

    async def generate_one_prompt(
        client: AsyncOpenAI, *, sem: asyncio.Semaphore, prompt: str, idx: str
    ) -> list[dict[str, Any]]:
        messages = [{"role": "user", "content": prompt}]

        choices = []
        async def _call(n: int):
            try:
                async with sem:
                    res = await client.chat.completions.create(
                        model=args.model_ref,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        n=n,
                    )
                choices.extend(res.choices)
            except Exception:
                logger.exception("Error generating completion for idx={}", idx)

        if not send_one_request_per_sample:
            await _call(args.n_samples)
        else:
            for _ in range(args.n_samples):
                await _call(1)

        out_rows: list[dict[str, Any]] = []
        for sample_idx, ch in enumerate(choices):
            content = ""
            if ch.message is not None and ch.message.content is not None:
                content = ch.message.content
            out_rows.append({
                "idx": idx,
                "sample_idx": sample_idx,
                "prompt": prompt,
                "response": content,
            })
        return out_rows

    async def run_all() -> None:
        client_kwargs: dict[str, Any] = {"timeout": timeout_seconds}
        if api_base_url is not None:
            client_kwargs["base_url"] = api_base_url
        if getenv("OPENAI_API_KEY") is None:
            logger.warning("OPENAI_API_KEY is not set, using a dummy key.")
            environ["OPENAI_API_KEY"] = "."
        client = AsyncOpenAI(**client_kwargs)
        sem = asyncio.Semaphore(concurrency)

        tasks = [asyncio.create_task(
            generate_one_prompt(client, sem=sem, prompt=r["prompt"], idx=str(r["idx"]))
        ) for r in rows]

        with open(out_f, "w") as fh:
            for t in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
                one_prompt_rows = await t
                for out_row in one_prompt_rows:
                    fh.write(json.dumps(out_row) + "\n")

    logger.info("Starting generation via OpenAI-compatible API...")
    asyncio.run(run_all())
    logger.success("Wrote: {}", cmd.cwd_rel(out_f))


@app.command()
def verify_from_generate(
    use_prebuilt_executor_images: Annotated[bool, typer.Option(
        help='Verify the code using prebuilt executor images from a public registry.'
    )] = False,
    executor_image_override: Annotated[str | None, typer.Option(
        help='Override the executor image name.'
    )] = None,
    timeout_seconds: Annotated[int, typer.Option(
        help='Per-solution timeout, seconds'
    )] = 15,
    num_concurrent: Annotated[int, typer.Option(
        help='Max concurrent container executions'
    )] = 20,
) -> None:
    """
    Grade the generations using executor containers.
    """


    from dockerinator.agnostics_cfg import get_container_tool
    from dockerinator.agnostics_resolve_container_tool_command import resolve_container_tool_command

    def _get_image_name(lang: str) -> str:
        if use_prebuilt_executor_images:
            return proglangs.PREBUILT_IMAGE_NAMES[lang]
        return proglangs.IMAGE_NAMES[lang]

    def _decompress_lcb_private_tests(text: str):
        return json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(text.encode('utf-8'))))
        )

    args = ANALYSIS_ARGS
    assert args is not None

    wd = analysis_wd_from_args(args)

    CmdExtractAnswersInstance.extract_answers(
        workdir=wd,
        input_path=None,
        force=True,
    )

    answers_path = wd / 'answers' / 'result.jsonl'
    verify_dir = wd / 'alt-verify'
    verify_path = verify_dir / 'result.jsonl'
    assert answers_path.exists()
    verify_dir.mkdir(parents=True, exist_ok=True)

    container_tool = get_container_tool()
    container_image_name = executor_image_override or _get_image_name(args.lang)
    container_cmd = resolve_container_tool_command(
        tool=container_tool,
        executor_image_name=container_image_name,
    )

    import datasets
    problems = datasets.load_dataset('nuprl/Ag-LiveCodeBench-X', split='test')
    tests_by_id: dict[str, str] = {p['question_id']: p['private_test_cases'] for p in problems}
    del problems

    from bounded_subprocess.bounded_subprocess_async import run as bounded_run

    async def execute_row(row: dict) -> dict:
        idx = row.get('idx')
        sample_idx = row.get('sample_idx')
        answer = row.get('answer')

        base_out: dict[str, Any] = { 'idx': idx }
        if sample_idx is not None:
            base_out['sample_idx'] = sample_idx

        if not answer:
            return {
                **base_out,
                'result': 'fail:no-answer',
                'raw_exit_code': -1,
                'raw_stdout': '',
                'raw_stderr': '',
            }

        key: str = str(idx or '')
        private_tests_compressed = tests_by_id.get(key)
        if private_tests_compressed is None:
            logger.error('No tests found when processing idx={}, sample_idx={}', idx, sample_idx)
            return {
                **base_out,
                'result': 'fail:no-tests',
                'raw_exit_code': -1,
                'raw_stdout': '',
                'raw_stderr': '',
            }

        result = await bounded_run(
            container_cmd,
            timeout_seconds=timeout_seconds,
            stdin_data=json.dumps({
                'code': answer,
                'timeout_s': timeout_seconds,
                'test_cases': _decompress_lcb_private_tests(private_tests_compressed),
            }),
            stdin_write_timeout=300,
        )

        raw_fields = {
            **base_out,
            'raw_exit_code': result.exit_code,
            'raw_stdout': result.stdout,
            'raw_stderr': result.stderr,
        }
        if result.exit_code != 0:
            return {**raw_fields, 'result': 'fail'}
        try:
            parsed = json.loads(result.stdout)
            return {**raw_fields, **parsed}
        except json.JSONDecodeError:
            return {**raw_fields, 'result': 'fail'}

    async def run_all() -> None:
        from abstractions.storage import map_by_key_jsonl_file
        from tqdm.auto import tqdm

        pbar = tqdm(desc='Executing')
        await map_by_key_jsonl_file(
            answers_path,
            verify_path,
            f=execute_row,
            key='answer',
            keep_columns=['idx', 'sample_idx'],
            on_error='raise',
            num_concurrent=num_concurrent,
            progress=lambda _: pbar.update(1),
        )

    logger.info('Starting verification...')
    asyncio.run(run_all())
    logger.success('Wrote: {}', verify_path)


@app.command()
def alt_verify_from_generate(
    max_workers: int | None = None,
    executor_image_override: str | None = None,
    resume: bool = False,
) -> None:
    """
    Alternative command for grading the generations;
    it runs executor containers more uniformly with the rest of the codebase but may be less performant.
    """

    args = ANALYSIS_ARGS
    assert args is not None

    wd = analysis_wd_from_args(args)

    CmdExtractAnswersInstance.extract_answers(
        workdir=wd,
        input_path=None,
        force=True,
    )

    import datasets
    logger.info('Building the examples lookup table...')
    problems = datasets.load_dataset('nuprl/Ag-LiveCodeBench-X', split='test')
    tests_by_id = {p['question_id']: p['private_test_cases'] for p in problems}
    del problems
    logger.info('Built the examples lookup table.')

    @lru_cache(maxsize=60)
    def decompress_lcb_private_tests(text: str):
        """
        LiveCodeBench compresses its private tests because they are enormous (8GB
        when we write our 499 problem subset to disk).
        """
        return json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(text.encode('utf-8'))))
        )

    def _examples_lookup_fn(idx: str) -> None|list[dict]:
        found = tests_by_id.get(idx)
        if found is None:
            return None
        decompressed = decompress_lcb_private_tests(found)
        return decompressed

    resolved_resume = bool(getenv('AGNOSTICS_RESUME')) or resume
    if resolved_resume:
        logger.info(
            'Resuming from the existing output file.'
            ' Note: we still decode the test cases before skipping the item.'
        )

    if os.getenv('AGNOSTICS_NO_CONTAINER_IO_LOGS') is None:
        os.environ['AGNOSTICS_NO_CONTAINER_IO_LOGS'] = '1'
    verify_cmd = proglangs.VERIFY_CMDS[args.lang]
    verify_cmd._verify_answers_with_examples_lookup_fn(
        examples_lookup_fn=_examples_lookup_fn,
        workdir=wd,
        max_workers=max_workers,
        executor_image_override=executor_image_override,
        force=True,
        resume=resolved_resume,
    )


if __name__ == '__main__':
    app()