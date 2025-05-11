"""
Code for benchmarking a local model trained with GRPO on Ag-Codeforces-X.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from ..codeforces_cots import proglangs

app = typer.Typer()


@dataclass
class AnalysisArgs:
    chkp_path: Path
    lang: str
    temperature: float
    temperature_str: str
    n_samples: int


MODEL_ENV_CFG = 'qwen3-4B+max-tokens-5k'

ANALYSIS_ARGS: AnalysisArgs | None = None


def get_model_env_cfg_key(
    model_env_cfg: str,
) -> str:
    from ..codeforces_cots.common import vllm_configs
    return vllm_configs.ENV_CONFIGS[model_env_cfg].model_config.config_key


def analysis_wd_from_run_ref(
    run_ref: Path,
    checkpoint_id: str,
    lang: str,
    temperature_str: str,
    n_samples: int,
) -> Path:
    clean_temperature_str = temperature_str.replace(".", "p")
    return run_ref/'analysis-codeforces-cots'/checkpoint_id/f'{lang}__n{n_samples}_tmpr{clean_temperature_str}'


def analysis_wd_from_args(
    args: AnalysisArgs,
) -> Path:
    return analysis_wd_from_run_ref(
        run_ref=args.chkp_path.parent,
        checkpoint_id=args.chkp_path.name,
        lang=args.lang,
        temperature_str=args.temperature_str,
        n_samples=args.n_samples,
    )


@app.callback()
def set_shared_analysis_args(
    lang: Annotated[str, typer.Option(help='The PL to generate code for.')],
    temperature: Annotated[str, typer.Option(help='The temperature to use for generation.')],
    n_samples: Annotated[int, typer.Option(help='The number of samples to generate.')],
    checkpoint_ref: Annotated[str, typer.Option(help=(
        'A ref (path) to the checkpoint to generate code with.'
    ))],
) -> None:
    """
    Sets analysis arguments shared between the steps.

    Some of the arguments are shared just to determine the working directory.
    """
    global ANALYSIS_ARGS

    assert not ANALYSIS_ARGS

    if lang not in proglangs.PROGLANGS:
        raise typer.BadParameter(f'--lang must be one of: {", ".join(proglangs.PROGLANGS)}')

    chkp_path = Path(checkpoint_ref)
    if not chkp_path.is_dir():
        raise typer.BadParameter(f'--checkpoint-ref must be a path to a checkpoint.')
    if not list(chkp_path.glob('*.safetensors')):
        raise typer.BadParameter(f'--checkpoint-ref points to a directory without any .safetensors files.')

    temperature_str = temperature
    try:
        temperature_float = float(temperature_str)
    except ValueError:
        raise typer.BadParameter(f'--temperature must be a float.')
    assert n_samples > 0

    ANALYSIS_ARGS = AnalysisArgs(
        chkp_path=chkp_path,
        lang=lang,
        temperature=temperature_float,
        temperature_str=temperature_str,
        n_samples=n_samples,
    )


@app.command()
def generate(
    batch_size: Annotated[int, typer.Option(help='The batch size to use for generation.')],
    force: bool = False,
) -> None:
    args = ANALYSIS_ARGS
    assert args is not None

    generate_cmd = proglangs.GENERATE_CMDS[args.lang]
    generate_cmd.generate(
        model_env_cfg=MODEL_ENV_CFG,
        model_ref=args.chkp_path,
        batch_size=batch_size,
        temperature=args.temperature,
        n_samples=args.n_samples,
        workdir=analysis_wd_from_args(args),
        force=force,
    )


@app.command()
def verify_from_generate() -> None:
    args = ANALYSIS_ARGS
    assert args is not None

    verify_cmd = proglangs.VERIFY_CMDS[args.lang]
    verify_cmd.verify_from_generate(
        workdir=analysis_wd_from_args(args),
        force=True,
    )


if __name__ == '__main__':
    app()
