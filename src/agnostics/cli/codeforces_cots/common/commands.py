from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Annotated, Any, Callable, Collection, Iterable, Iterator
from itertools import batched, islice
import io

from loguru import logger
import typer
from tqdm import tqdm

from agnostics.schema import solutions_py
from agnostics.util.code_finder import thinks_end_rx, find_final_answer_block
from dockerinator.run_in_containers import ExecutionResultRow
from ..preprocess_solutions_py import SolutionsRow
from .. import train_split, test_split, validation_split
from ... import cmd


STANDARD_PROMPT_PREFIX = '''\
Your task is to solve a competitive programming problem.

# Problem
'''

STANDARD_PROMPT_INPUT_FORMAT_HEADER = '''\

# Input Format
'''

STANDARD_PROMPT_OUTPUT_FORMAT_HEADER = '''\

# Output Format
'''

STANDARD_PROMPT_EXAMPLES_HEADER = '''\

# Examples
'''

STANDARD_PROMPT_ONE_EXAMPLE_TEMPLATE = '''\
# Example {i}
Input:
```
{input}
```

Output:
```
{output}
```
'''

STANDARD_PROMPT_NOTES_HEADER = '''\

# Notes
'''

STANDARD_PROMPT_SUFFIX = '''\

# Instructions
Provide a complete, fully-implemented solution to the problem. \
Make sure your solution generalizes to other test cases.
'''

ALT_PROMPT_PREFIX = ''

ALT_PROMPT_INPUT_FORMAT_HEADER = '''\

**Stdin specification**
'''

ALT_PROMPT_OUTPUT_FORMAT_HEADER = '''\

**Stdout specification**
'''

ALT_PROMPT_EXAMPLES_HEADER = '''\

**Samples**
'''

ALT_PROMPT_ONE_EXAMPLE_TEMPLATE = '''\
**Sample {i}**
Standard input:
```
{input}
```

Standard output:
```
{output}
```
'''

ALT_PROMPT_NOTES_HEADER = '''\

**Details**
'''

ALT_PROMPT_SUFFIX = '''\

**Task**
'''


class AbstractCmdMakePrompts():
    def __init__(
        self,
        file_attr: str,
    ):
        self.file_attr = file_attr

    @abstractmethod
    def build_prompt(
        self,
        in_row: SolutionsRow,
        allow_omitting_only_example: bool = False,
    ) -> str:
        ...

    @abstractmethod
    def build_variant_prompt(
        self,
        rng: random.Random,
        in_row: SolutionsRow,
    ) -> str:
        ...

    def make_prompts(
        self,
        input_file: Path | None = None,
        input_split: str | None = None,
        range_start: int = 0,
        range_size: int = -1,
        skip_invalid_rows: bool = False,
        workdir: Path | None = None,
    ):
        """
        Make prompts, by default from the test split of agnostics-codeforces-cots.
        """
        valid_input_splits = ('train', 'validation', 'test', 'train-xl-varprompt')
        if input_split and input_split not in valid_input_splits:
            raise typer.BadParameter(f'--input-split is {input_split!r}, but must be one of {", ".join(valid_input_splits)}')

        if sum(map(bool, (input_file, input_split))) > 1:
            raise typer.BadParameter('Only one of --input-file or --input-split should be provided')


        ctx = cmd.start(
            self.file_attr,
            out_d=workdir,
            sub_out_d='make-prompts',
            force=True,
        )
        out_f = ctx.out_d/'result.jsonl'

        def _gen_data_rows():
            nonlocal input_split
            start = range_start
            end = start + range_size if range_size != -1 else None
            if not input_file:
                if not input_split:
                    input_split = 'test'
                if input_split == 'train':
                    gen_ds_rows = train_split.gen_from_hf()
                elif input_split == 'validation':
                    gen_ds_rows = validation_split.gen_from_hf()
                elif input_split == 'test':
                    gen_ds_rows = test_split.gen_from_hf()
                elif input_split == 'train-xl-varprompt':
                    gen_ds_rows = train_split.gen_xl_from_hf()
                else:
                    raise ValueError(f'Unexpected value: {input_split=!r}')
            else:
                gen_ds_rows = cmd.ser.model_jsonl_streamf(SolutionsRow, input_file)
            if skip_invalid_rows:
                gen_ds_rows = (r for r in gen_ds_rows if len(r.examples) > 1)
            return islice(gen_ds_rows, start, end)

        if input_split != 'train-xl-varprompt':
            def build_prompt(in_row: SolutionsRow) -> str:
                return self.build_prompt(in_row)
        else:
            rng = random.Random(42)
            def build_prompt(in_row: SolutionsRow) -> str:
                return self.build_variant_prompt(rng, in_row)

        def gen_output_rows(ds_rows: Iterable[SolutionsRow]) -> Iterator[dict]:
            for r in ds_rows:
                r = { 'idx': r.idx, 'prompt': build_prompt(r), }
                yield r

        cmd.ser.jsonl_dumpf(gen_output_rows(_gen_data_rows()), out_f)
        logger.success('Wrote: {}', cmd.cwd_rel(out_f))

    def add_commands(self, app: typer.Typer) -> None:
        app.command()(self.make_prompts)


class CmdStandardMakePrompts(AbstractCmdMakePrompts):
    def __init__(
        self,
        file_attr: str,
        /,
        prompt_pl_suffix: str,
        prompt_prefix: str = STANDARD_PROMPT_PREFIX,
        prompt_input_format_header: str = STANDARD_PROMPT_INPUT_FORMAT_HEADER,
        prompt_output_format_header: str = STANDARD_PROMPT_OUTPUT_FORMAT_HEADER,
        prompt_examples_header: str = STANDARD_PROMPT_EXAMPLES_HEADER,
        prompt_one_example_template: str = STANDARD_PROMPT_ONE_EXAMPLE_TEMPLATE,
        prompt_notes_header: str = STANDARD_PROMPT_NOTES_HEADER,
        prompt_suffix: str = STANDARD_PROMPT_SUFFIX,
    ):
        super().__init__(file_attr)
        self.prompt_prefix = prompt_prefix
        self.prompt_input_format_header = prompt_input_format_header
        self.prompt_output_format_header = prompt_output_format_header
        self.prompt_examples_header = prompt_examples_header
        self.prompt_one_example_template = prompt_one_example_template
        self.prompt_notes_header = prompt_notes_header
        self.prompt_suffix = prompt_suffix
        self.prompt_pl_suffix = prompt_pl_suffix

    def build_prompt(
        self,
        in_row: SolutionsRow,
        allow_omitting_only_example: bool = False,
    ) -> str:
        b = io.StringIO()
        def put(*objects: str):
            print(*objects, sep='', end='', file=b)
        put(self.prompt_prefix, in_row.problem_statement, '\n')
        if in_row.input_format:
            put(self.prompt_input_format_header, in_row.input_format, '\n')
        if in_row.output_format:
            put(self.prompt_output_format_header, in_row.output_format, '\n')
        if len(in_row.examples) > 1:
            put(self.prompt_examples_header, '\n')
            for i, e in enumerate(in_row.examples[:-1], 1):
                if i > 1:
                    put('\n')
                put(self.prompt_one_example_template.format(i=i, input=e.input, output=e.output))
        else:
            assert allow_omitting_only_example, f'a row has <2 examples but {allow_omitting_only_example=}'
        if in_row.problem_notes:
            put(self.prompt_notes_header, in_row.problem_notes, '\n')
        put(self.prompt_suffix)
        put(self.prompt_pl_suffix)

        return b.getvalue()

    def build_variant_prompt(
        self,
        rng: random.Random,
        in_row: SolutionsRow,
    ) -> str:
        if len(in_row.examples) > 1:
            if rng.random() < 0.3:
                return self.build_prompt(in_row)
            elif rng.random() < 0.5:
                stored_prefix = self.prompt_prefix
                stored_input_format_header = self.prompt_input_format_header
                stored_output_format_header = self.prompt_output_format_header
                stored_examples_header = self.prompt_examples_header
                stored_one_example_template = self.prompt_one_example_template
                stored_notes_header = self.prompt_notes_header
                stored_suffix = self.prompt_suffix

                self.prompt_prefix = ALT_PROMPT_PREFIX
                self.prompt_input_format_header = ALT_PROMPT_INPUT_FORMAT_HEADER
                self.prompt_output_format_header = ALT_PROMPT_OUTPUT_FORMAT_HEADER
                self.prompt_examples_header = ALT_PROMPT_EXAMPLES_HEADER
                self.prompt_one_example_template = ALT_PROMPT_ONE_EXAMPLE_TEMPLATE
                self.prompt_notes_header = ALT_PROMPT_NOTES_HEADER
                self.prompt_suffix = ALT_PROMPT_SUFFIX
                res = self.build_prompt(in_row)

                self.prompt_prefix = stored_prefix
                self.prompt_input_format_header = stored_input_format_header
                self.prompt_output_format_header = stored_output_format_header
                self.prompt_examples_header = stored_examples_header
                self.prompt_one_example_template = stored_one_example_template
                self.prompt_notes_header = stored_notes_header
                self.prompt_suffix = stored_suffix

                return res

        b = io.StringIO()
        def put(*objects: str):
            print(*objects, sep='', end='', file=b)
        put(in_row.problem_statement, '\n')
        put(in_row.input_format, '\n')
        put(in_row.output_format, '\n')
        if len(in_row.examples) > 1 and rng.random() < 0.5:
            put(ALT_PROMPT_EXAMPLES_HEADER, '\n')
            for i, e in enumerate(in_row.examples[:-1], 1):
                if i > 1:
                    put('\n')
                put(ALT_PROMPT_ONE_EXAMPLE_TEMPLATE.format(i=i, input=e.input, output=e.output))
        if in_row.problem_notes:
            put(ALT_PROMPT_NOTES_HEADER, in_row.problem_notes, '\n')
        put('\n', self.prompt_pl_suffix)

        return b.getvalue()

class CmdGenerate():
    def __init__(
        self,
        file_attr: str,
    ):
        self._file_attr = file_attr

    def generate(
        self,
        model_env_cfg: Annotated[str|None, typer.Option(
            help=(
                'The name of a model-environment config defined in `vllm_configs.py`.'
                ' Pass a bad name (eg `help`) to see the options.'
            )
        )] = None,
        model_ref: Annotated[str|None, typer.Option(
            help='vLLM model reference.'
        )] = None,
        input_file: Annotated[Path | None, typer.Option(
            help='The file with the prompts to generate from. Defaults to the output of `make_prompts`.'
        )] = None,
        range_start: int = 0,
        range_size: int = -1,
        temperature: Annotated[float | None, typer.Option(
            help='Override the default temperature.'
        )] = None,
        batch_size: int = 1,
        n_samples: Annotated[int | None, typer.Option(
            help='Override the default number of samples to generate.',
            min=1,
        )] = None,
        max_tokens: Annotated[int | None, typer.Option(
            help='Override the default max output tokens.',
            min=1,
        )] = None,
        workdir: Annotated[Path | None, typer.Option(
            help=(
                'The root output dir for analyzing the generations. Defaults to a dir named after the model.'
                ' The output will be written to a subdir, as expected by other commands.'
            )
        )] = None,
        force: bool = False,
    ):
        """
        Generates model's responses to prompts from the input file.
        The model is specified by either --model-env-cfg or --model-ref.

        If both are provided, the model specified by --model-ref will override
        the default model associated with --model-env-cfg, but the rest of the
        configuration will be the same.
        """
        if not (bool(model_env_cfg) or bool(model_ref)):
            raise typer.BadParameter('Either --model-env-cfg or --model-ref must be provided.')
        elif model_env_cfg and model_ref:
            logger.info(
                'Received both --model-env-cfg and --model-ref; '
                'the former will be used to configure the latter.'
            )

        from . import vllm_configs, vllm_facade
        from ._gen_batch_output_rows import gen_batch_output_rows

        if workdir is None:
            if model_ref:
                sub_out_d = model_ref.replace('/', '_').replace('.', 'p')
            else:
                assert model_env_cfg
                try:
                    env_config = vllm_configs.ENV_CONFIGS[model_env_cfg]
                except KeyError:
                    raise typer.BadParameter(f'--model-env-cfg must be one of:\n{"\n".join(vllm_configs.ENV_CONFIGS.keys())}')
                sub_out_d = env_config.model_config.config_key

            sub_out_d = Path(sub_out_d)/'generate'
            ctx = cmd.start(
                self._file_attr,
                sub_out_d=sub_out_d,
                force=force,
            )
            out_d = ctx.out_d
            assert len(sub_out_d.parts) == 2
            default_workdir_root = ctx.out_d/'../..'
        else:
            out_d = workdir/'generate'
            if not force and out_d.exists():
                logger.error('Output dir already exists: {}', cmd.cwd_rel(out_d))
                logger.error('Use --force to overwrite an existing output dir.')
                raise typer.Exit(1)
            out_d.mkdir(parents=True, exist_ok=force)
            cmd.start_no_ctx(out_d)

            ctx = cmd.start(
                self._file_attr,
                readonly=True,
            )
            default_workdir_root = ctx.out_d
        logger.info('Starting...')

        resolved_data_f = input_file or default_workdir_root/'make-prompts/result.jsonl'

        out_config_f = out_d/'config.json'
        out_f = out_d/'result.jsonl'

        cmd.ser.json_dumpf(cmd.typecheck_jsonable({
            'env_name': model_env_cfg,
            'resolved_data_f': str(resolved_data_f),
            'range_start': range_start,
            'range_size': range_size,
            'temperature': temperature,
            'batch_size': batch_size,
            'n_samples': n_samples,
            'max_tokens': max_tokens,
        }), out_config_f)
        logger.info('Wrote step config: {}', cmd.cwd_rel(out_config_f))

        start = range_start
        end = range_start+range_size if range_size >= 0 else -1
        data_size = range_size
        if range_size < 0:
            data_size = sum(1 for _ in cmd.ser.jsonl_streamf(resolved_data_f)) - start

        if model_ref:
            kwargs: dict[str, Any]
            if model_env_cfg:
                env_cfg = vllm_configs.ENV_CONFIGS[model_env_cfg]
                kwargs = env_cfg.as_model_handle_kwargs()
            else:
                kwargs = {
                    'default_sampling_params': vllm_facade.SamplingParams(
                        max_tokens=vllm_configs.REASONABLE_MAX_TOKENS
                    ),
                }
            kwargs['model_ref'] = model_ref
            model = vllm_facade.make_model_handle(**kwargs)
        else:
            assert model_env_cfg
            model = vllm_configs.model_from_env_cfg_name(model_env_cfg)

        sp = None
        if temperature or n_samples or max_tokens:
            sp = model.default_sampling_params.clone()
            if temperature:
                sp.temperature = temperature
            if n_samples:
                sp.n = n_samples
            if max_tokens:
                sp.max_tokens = max_tokens
        resolved_n_samples = n_samples or model.default_sampling_params.n

        logger.info('Model loaded, starting generation...')
        out_rows_gen = (
            r
            for in_rows_batch in batched(
                cmd.ser.jsonl_streamf(resolved_data_f, start=start, end=end),
                batch_size,
            )
            for r in gen_batch_output_rows(model, in_rows_batch, override_sampling_params=sp)
        )
        if not os.getenv('AGNOSTICS_USE_VLLM_TQDM'):
            out_rows_gen = tqdm(out_rows_gen, total=data_size*resolved_n_samples)
        cmd.ser.jsonl_dumpf(out_rows_gen, out_f)
        logger.success('Wrote: {}', cmd.cwd_rel(out_f))

    def add_commands(self, app: typer.Typer) -> None:
        app.command()(self.generate)

    def _generate_from_input_row_generator(
        self,
        input_row_gen: Iterable[dict],
        gen_size: int,
        model_env_cfg: Annotated[str|None, typer.Option(
            help=(
                'The name of a model-environment config defined in `vllm_configs.py`.'
                ' Pass a bad name (eg `help`) to see the options.'
            )
        )] = None,
        model_ref: Annotated[str|None, typer.Option(
            help='vLLM model reference.'
        )] = None,
        temperature: Annotated[float | None, typer.Option(
            help='Override the default temperature.'
        )] = None,
        batch_size: int = 1,
        n_samples: Annotated[int | None, typer.Option(
            help='Override the default number of samples to generate.',
            min=1,
        )] = None,
        max_tokens: Annotated[int | None, typer.Option(
            help='Override the default max output tokens.',
            min=1,
        )] = None,
        workdir: Annotated[Path | None, typer.Option(
            help=(
                'The root output dir for analyzing the generations. Defaults to a dir named after the model.'
                ' The output will be written to a subdir, as expected by other commands.'
            )
        )] = None,
        force: bool = False,
    ):
        """
        Generates model's responses to prompts from the input file.
        The model is specified by either --model-env-cfg or --model-ref.

        If both are provided, the model specified by --model-ref will override
        the default model associated with --model-env-cfg, but the rest of the
        configuration will be the same.
        """
        if not (bool(model_env_cfg) or bool(model_ref)):
            raise typer.BadParameter('Either --model-env-cfg or --model-ref must be provided.')
        elif model_env_cfg and model_ref:
            logger.info(
                'Received both --model-env-cfg and --model-ref; '
                'the former will be used to configure the latter.'
            )

        from . import vllm_configs, vllm_facade
        from ._gen_batch_output_rows import gen_batch_output_rows

        if workdir is None:
            if model_ref:
                sub_out_d = model_ref.replace('/', '_').replace('.', 'p')
            else:
                assert model_env_cfg
                try:
                    env_config = vllm_configs.ENV_CONFIGS[model_env_cfg]
                except KeyError:
                    raise typer.BadParameter(f'--model-env-cfg must be one of:\n{"\n".join(vllm_configs.ENV_CONFIGS.keys())}')
                sub_out_d = env_config.model_config.config_key

            sub_out_d = Path(sub_out_d)/'generate'
            ctx = cmd.start(
                self._file_attr,
                sub_out_d=sub_out_d,
                force=force,
            )
            out_d = ctx.out_d
            assert len(sub_out_d.parts) == 2
            default_workdir_root = ctx.out_d/'../..'
        else:
            out_d = workdir/'generate'
            if not force and out_d.exists():
                logger.error('Output dir already exists: {}', cmd.cwd_rel(out_d))
                logger.error('Use --force to overwrite an existing output dir.')
                raise typer.Exit(1)
            out_d.mkdir(parents=True, exist_ok=force)
            cmd.start_no_ctx(out_d)

            ctx = cmd.start(
                self._file_attr,
                readonly=True,
            )
            default_workdir_root = ctx.out_d
        logger.info('Starting...')

        out_config_f = out_d/'config.json'
        out_f = out_d/'result.jsonl'

        cmd.ser.json_dumpf(cmd.typecheck_jsonable({
            'env_name': model_env_cfg,
            'temperature': temperature,
            'batch_size': batch_size,
            'n_samples': n_samples,
            'max_tokens': max_tokens,
        }), out_config_f)
        logger.info('Wrote step config: {}', cmd.cwd_rel(out_config_f))

        if model_ref:
            kwargs: dict[str, Any]
            if model_env_cfg:
                env_cfg = vllm_configs.ENV_CONFIGS[model_env_cfg]
                kwargs = env_cfg.as_model_handle_kwargs()
            else:
                kwargs = {
                    'default_sampling_params': vllm_facade.SamplingParams(
                        max_tokens=vllm_configs.REASONABLE_MAX_TOKENS
                    ),
                }
            kwargs['model_ref'] = model_ref
            model = vllm_facade.make_model_handle(**kwargs)
        else:
            assert model_env_cfg
            model = vllm_configs.model_from_env_cfg_name(model_env_cfg)

        sp = None
        if temperature or n_samples or max_tokens:
            sp = model.default_sampling_params.clone()
            if temperature:
                sp.temperature = temperature
            if n_samples:
                sp.n = n_samples
            if max_tokens:
                sp.max_tokens = max_tokens
        resolved_n_samples = n_samples or model.default_sampling_params.n

        logger.info('Model loaded, starting generation...')
        out_rows_gen = (
            r
            for in_rows_batch in batched(
                input_row_gen,
                batch_size,
            )
            for r in gen_batch_output_rows(model, in_rows_batch, override_sampling_params=sp)
        )
        cmd.ser.jsonl_dumpf(
            tqdm(out_rows_gen, total=gen_size*resolved_n_samples),
            out_f,
        )
        logger.success('Wrote: {}', cmd.cwd_rel(out_f))


def gen_answer_rows(
    dep_responses_rows: Iterable[dict],
) -> Iterator[dict]:
    for in_r in dep_responses_rows:
        idx = in_r['idx']
        response = in_r['response']

        r = { 'idx': idx }
        if 'sample_idx' in in_r:
            r['sample_idx'] = in_r['sample_idx']

        issues: list[str] = []
        r['issues'] = ''
        r['answer'] = None

        def gen_result():
            r['issues'] = ','.join(issues)
            yield r

        proper_response_offset = 0

        code = find_final_answer_block(response, proper_response_offset, answer_must_be_valid_python=False)
        if not code:
            issues.append('no-answer')
            yield from gen_result()
            continue

        r['answer'] = code
        yield from gen_result()


class CmdExtractAnswers():
    def extract_answers(
        self,
        workdir: Annotated[Path|None, typer.Option(
            help='The workdir with input/output dirs.'
        )] = None,
        input_path: Annotated[Path|None, typer.Option(
            help='The file with the responses to extract answers from (or parent dir of `result.jsonl`). Overrides --workdir.'
        )] = None,
        output_dir: Annotated[Path|None, typer.Option(
            help='The directory to write the output to. Overrides --workdir.'
        )] = None,
        allow_malformed_lines: Annotated[bool, typer.Option(
            help='Allow malformed lines in the input file.'
        )] = False,
        force: bool = False,
    ):
        """
        Extract answers from the generated output.
        """
        if workdir is None and input_path is None and output_dir is None:
            raise typer.BadParameter('A combination of --workdir / --input-path / --output-dir is required. See --help.')

        if input_path is None:
            if workdir is None:
                raise typer.BadParameter('Either --workdir or --input-path is required.')
            input_path = workdir/'generate/result.jsonl'
        if input_path.is_dir():
            input_path = input_path/'result.jsonl'
        if not input_path.exists():
            raise typer.BadParameter(f'expected to exist: {input_path}')

        if output_dir is None:
            if workdir is None:
                raise typer.BadParameter('Either --workdir or --output-dir is required.')
            output_dir = workdir/'answers'
        if not force and output_dir.exists() and any(output_dir.iterdir()):
            raise typer.BadParameter(f'--force required to overwrite output dir: {output_dir}')
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd.start_no_ctx(out_d=None)

        out_f = output_dir/'result.jsonl'

        cmd.ser.jsonl_dumpf(
            gen_answer_rows(
                cmd.ser.jsonl_streamf(input_path, allow_malformed_lines=allow_malformed_lines),
            ),
            out_f,
        )
        logger.success('Wrote: {}', cmd.cwd_rel(out_f))

    def add_commands(self, app: typer.Typer) -> None:
        app.command()(self.extract_answers)


CmdExtractAnswersInstance = CmdExtractAnswers()


@dataclass
class VerifierItem():
    key: str
    io_examples: list[solutions_py.IOExample]
    code: str


class AbstractCmdVerifyAnswers(metaclass=ABCMeta):
    @abstractmethod
    def run_answer_verifier(
        self,
        answer_rows: Iterable[dict],
        ds_by_idx: dict[int, solutions_py.SolutionsRow],
        out_report_f: Path,
        timeout_s: int | None = None,
        max_workers: int | None = None,
    ) -> None:
        ...

    def verify_answers(
        self,
        workdir: Annotated[Path|None, typer.Option(
            help='The workdir with input/output dirs.'
        )] = None,
        source_ds_file: Annotated[Path|None, typer.Option(
            help=(
                'The source dataset used to make problems for which the answers were generated.'
                ' Defaults to the test split of the `agnostics-codeforces-cots` HF dataset.'
            )
        )] = None,
        input_path: Annotated[Path|None, typer.Option(
            help=(
                'The file with the answers to verify.'
                ' For convenience, `result.jsonl` is appended if this is a dir.'
                ' Overrides --workdir.'
            )
        )] = None,
        output_dir: Annotated[Path|None, typer.Option(
            help='The directory to write the output to. Overrides --workdir.'
        )] = None,
        timeout_s: int | None = None,
        force: bool = False,
        max_workers: int | None = None,
    ):
        """
        Verify previously extracted answers.

        It reads the expected answers from the source dataset (specified by the --source-ds-file).
        """
        if workdir is None and input_path is None and output_dir is None:
            raise typer.BadParameter('A combination of --workdir / --input-path / --output-dir is required. See --help.')

        if input_path is None:
            if workdir is None:
                raise typer.BadParameter('Either --workdir or --input-path is required.')
            input_path = workdir/'answers/result.jsonl'
        if input_path.is_dir():
            input_path = input_path/'result.jsonl'
        if not input_path.exists():
            raise typer.BadParameter(f'expected to exist: {input_path}')

        if output_dir is None:
            if workdir is None:
                raise typer.BadParameter('Either --workdir or --output-dir is required.')
            output_dir = workdir/'verify'
        if not force and output_dir.exists() and any(output_dir.iterdir()):
            raise typer.BadParameter(f'--force required to overwrite output dir: {output_dir}')
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd.start_no_ctx(out_d=None)

        out_f = output_dir/'result.jsonl'

        if not source_ds_file:
            from .. import test_split
            ds_by_idx = {
                r.idx: r for r in test_split.gen_from_hf()
            }
        else:
            ds_by_idx = {
                r.idx: r for r in cmd.ser.model_jsonl_streamf(solutions_py.SolutionsRow, source_ds_file)
            }

        self.run_answer_verifier(
            answer_rows=cmd.ser.jsonl_streamf(input_path),
            ds_by_idx=ds_by_idx,
            out_report_f=out_f,
            timeout_s=timeout_s,
            max_workers=max_workers,
        )
        logger.success('Wrote: {}', cmd.cwd_rel(out_f))



    def describe_verify_results(
        self,
        input_files: list[Path],
    ):
        import pandas as pd

        cmd.start_no_ctx(out_d=None)

        print('file\tsuccess-rate')
        for input_file in input_files:
            df = pd.read_json(input_file, lines=True)
            score = (df['status'] == 'success').mean()
            print(f'{input_file}\t{score:.2%}')


    def verify_from_generate(
        self,
        workdir: Annotated[Path, typer.Option(
            help='The workdir with input/output dirs.'
        )],
        source_ds_file: Annotated[Path|None, typer.Option(
            help=(
                'The source dataset used to make problems for which the answers were generated.'
                ' Defaults to the test split of the `agnostics-codeforces-cots` HF dataset.'
            )
        )] = None,
        input_path: Annotated[Path|None, typer.Option(
            help=(
                'The file with the responses to extract answers from (or parent dir of `result.jsonl`).'
                ' Overrides --workdir.'
            )
        )] = None,
        timeout_s: int | None = None,
        force: bool = False,
        max_workers: int | None = None,
    ):
        """
        Equivalent to both extract_answers and verify_answers.
        """
        CmdExtractAnswersInstance.extract_answers(
            workdir=workdir,
            input_path=input_path,
            force=force,
        )
        self.verify_answers(
            workdir=workdir,
            source_ds_file=source_ds_file,
            force=force,
            timeout_s=timeout_s,
            max_workers=max_workers,
        )

    def add_commands(
        self,
        app: typer.Typer,
    ) -> None:
        app.command()( self.verify_answers )
        app.command()( self.verify_from_generate )
        app.command()( self.describe_verify_results )


    def _verify_answers_with_examples_lookup_fn(
        self,
        examples_lookup_fn: Callable[[str], None|dict],
        workdir: Annotated[Path|None, typer.Option(
            help='The workdir with input/output dirs.'
        )] = None,
        input_path: Annotated[Path|None, typer.Option(
            help=(
                'The file with the answers to verify.'
                ' For convenience, `result.jsonl` is appended if this is a dir.'
                ' Overrides --workdir.'
            )
        )] = None,
        output_dir: Annotated[Path|None, typer.Option(
            help='The directory to write the output to. Overrides --workdir.'
        )] = None,
        timeout_s: int | None = None,
        max_workers: int | None = None,
        executor_image_override: str | None = None,
        force: bool = False,
        resume: bool = False,
    ):
        """
        Verify previously extracted answers.

        It reads the expected answers from the source dataset (specified by the --source-ds-file).
        """
        if workdir is None and input_path is None and output_dir is None:
            raise typer.BadParameter('A combination of --workdir / --input-path / --output-dir is required. See --help.')

        if input_path is None:
            if workdir is None:
                raise typer.BadParameter('Either --workdir or --input-path is required.')
            input_path = workdir/'answers/result.jsonl'
        if input_path.is_dir():
            input_path = input_path/'result.jsonl'
        if not input_path.exists():
            raise typer.BadParameter(f'expected to exist: {input_path}')

        if output_dir is None:
            if workdir is None:
                raise typer.BadParameter('Either --workdir or --output-dir is required.')
            output_dir = workdir/'verify'
        if not force and output_dir.exists() and any(output_dir.iterdir()):
            raise typer.BadParameter(f'--force required to overwrite output dir: {output_dir}')
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd.start_no_ctx(out_d=None)

        out_f = output_dir/'result.jsonl'

        self._run_answer_verifier_with_examples_lookup_fn(
            answer_rows=cmd.ser.jsonl_loadf(input_path),
            examples_lookup_fn=examples_lookup_fn,
            out_report_f=out_f,
            timeout_s=timeout_s,
            max_workers=max_workers,
            executor_image_override=executor_image_override,
            resume=resume,
        )
        logger.success('Wrote: {}', cmd.cwd_rel(out_f))



from dockerinator import (
    ExecutionArgs,
    ItemAndId,
    AgnosticsItem,
    AgnosticsTestCase,
    run_agnostics_items,
)

class CmdVerifyAnswers(AbstractCmdVerifyAnswers):
    def __init__(
        self,
        executor_image_name: str,
    ):
        self._executor_image_name = executor_image_name

    def run_answer_verifier(
        self,
        answer_rows: Iterable[dict],
        ds_by_idx: dict[int, solutions_py.SolutionsRow],
        out_report_f: Path,
        timeout_s: int | None = None,
        max_workers: int | None = None,
    ) -> None:
        items: list[ItemAndId[AgnosticsItem]] = []
        answerless_item_keys: list[str] = []
        missing_problem_rows_num = 0
        for in_r in answer_rows:
            idx = in_r['idx']
            if 'sample_idx' in in_r:
                key = '/'.join((str(in_r[k]) for k in ('idx', 'sample_idx')))
            else:
                logger.warning('Assuming the input is in older format - missing field: sample_idx')
                key = str(idx)

            answer = in_r['answer']
            if not answer:
                answerless_item_keys.append(key)
                continue

            if idx not in ds_by_idx:
                logger.error('Problem row lookup in ds_by_idx failed (no examples?) for idx: {}', idx)
                missing_problem_rows_num += 1
                continue
            ds_r = ds_by_idx[idx]

            items.append(ItemAndId(id=key, item=AgnosticsItem(
                code=answer,
                test_cases=[AgnosticsTestCase(input=e.input, output=e.output) for e in ds_r.examples],
            )))

        io_log_dir = out_report_f.parent/'container-io-logs'
        io_log_dir.mkdir(parents=True, exist_ok=True)
        for f in io_log_dir.iterdir():
            f.unlink()

        run_agnostics_items(
            executor_image_name=self._executor_image_name,
            inputs=items,
            container_io_log_dir=io_log_dir,
            execution_args=ExecutionArgs(
                max_workers=max_workers,
                timeout_s=timeout_s,
                output_file=out_report_f,
            ),
        )

        with out_report_f.open('a') as fh:
            answerless_rows_gen = (
                ExecutionResultRow(key, 'fail:no-answer').to_jsonable()
                for key in answerless_item_keys
            )
            cmd.ser.jsonl_dumpfh( answerless_rows_gen, fh )

        if missing_problem_rows_num > 0:
            logger.error('Failing due to missing problem rows, count: {}', missing_problem_rows_num)
            raise typer.Exit(1)


    def _run_answer_verifier_with_examples_lookup_fn(
        self,
        answer_rows: Collection[dict],
        examples_lookup_fn: Callable[[str], None|dict],
        out_report_f: Path,
        timeout_s: int | None = None,
        max_workers: int | None = None,
        executor_image_override: str | None = None,
        resume: bool = False,
    ) -> None:
        answerless_item_keys: list[str] = []
        missing_problem_rows_num = 0
        def _gen_items():
            for in_r in answer_rows:
                idx = in_r['idx']
                if 'sample_idx' in in_r:
                    key = '/'.join((str(in_r[k]) for k in ('idx', 'sample_idx')))
                else:
                    logger.warning('Assuming the input is in older format - missing field: sample_idx')
                    key = str(idx)

                answer = in_r['answer']
                if not answer:
                    answerless_item_keys.append(key)
                    continue

                examples = examples_lookup_fn(idx)
                if examples is None:
                    logger.error('Problem row lookup in ds_by_idx failed (no examples?) for idx: {}', idx)
                    missing_problem_rows_num += 1
                    continue

                yield ItemAndId(id=key, item=AgnosticsItem(
                    code=answer,
                    test_cases=[AgnosticsTestCase(input=e['input'], output=e['output']) for e in examples],
                ))

        io_log_dir = out_report_f.parent/'container-io-logs'
        io_log_dir.mkdir(parents=True, exist_ok=True)
        for f in io_log_dir.iterdir():
            f.unlink()

        execution_args = ExecutionArgs(
            max_workers=max_workers,
            timeout_s=timeout_s,
            output_file=out_report_f,
        )

        executor_image = executor_image_override or self._executor_image_name

        io_log_dir.mkdir(parents=True, exist_ok=True)
        for f in io_log_dir.iterdir():
            f.unlink()

        from dockerinator.agnostics import AgnosticsContainerSupervisor, _run_supervisor_with_item_gen
        supervisor = AgnosticsContainerSupervisor(
            executor_image_name=executor_image,
            args=execution_args,
            io_log_dir=io_log_dir,
        )

        _run_supervisor_with_item_gen(
            executor_image_name=executor_image,
            item_gen=_gen_items(),
            gen_size=len(answer_rows),
            supervisor=supervisor,
            execution_args=execution_args,
            resume=resume,
        )

        with out_report_f.open('a') as fh:
            answerless_rows_gen = (
                ExecutionResultRow(key, 'fail:no-answer').to_jsonable()
                for key in answerless_item_keys
            )
            cmd.ser.jsonl_dumpfh( answerless_rows_gen, fh )

        if missing_problem_rows_num > 0:
            logger.error('Failing due to missing problem rows, count: {}', missing_problem_rows_num)
            raise typer.Exit(1)
