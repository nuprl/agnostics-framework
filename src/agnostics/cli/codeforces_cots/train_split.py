import random
from typing import Any, Iterator

from tqdm import tqdm
import typer
from loguru import logger

from . import preprocess_solutions_py, preprocess_checker_interactor, validation_split, test_split
from .validation_split import make_solutions_row
from .. import cmd

app = typer.Typer()


def gen_from_hf() -> Iterator[preprocess_solutions_py.SolutionsRow]:
    """
    A helper function for other modules: loads the train split from the HF dataset and yields the rows.
    """
    import datasets
    ds = datasets.load_dataset('nuprl/agnostics-codeforces-cots', split='train')

    for in_row in tqdm(ds, desc='Processing the train split'):
        yield make_solutions_row(in_row)


def gen_from_dep_file(
    start: int = 0,
    end: int = -1,
) -> Iterator[preprocess_solutions_py.SolutionsRow]:
    ctx = cmd.start(__file__, readonly=True)
    dep_f = ctx.out_d/'fresh-train-split.jsonl'
    if not dep_f.exists():
        logger.error('Dependency file does not exist: {}', dep_f)
        raise typer.Exit(1)

    yield from cmd.ser.model_jsonl_streamf(preprocess_solutions_py.SolutionsRow, dep_f, start, end)


def gen_xl_from_hf() -> Iterator[preprocess_solutions_py.SolutionsRow]:
    taken_indices = set(validation_split.gen_indices_from_hf())
    taken_indices.update(test_split.gen_indices_from_hf())

    wanted_indices = set(
        r['idx'] for r in preprocess_checker_interactor.gen_from_hf() if r['type'] == 'diff'
    )

    yield from (
        r for r in preprocess_solutions_py.gen_preprocessed_rows()
        if r.idx not in taken_indices and r.idx in wanted_indices
    )


def gen_indices_from_hf() -> Iterator[int]:
    for r in gen_from_hf():
        yield r.idx


def gen_indices_from_dep_file() -> Iterator[int]:
    for r in gen_from_dep_file():
        yield r.idx


@app.command()
def write_hf() -> None:
    """
    Writes the HF train split to a local file.
    """
    ctx = cmd.start(
        __file__,
        force=True,
    )
    out_f = ctx.out_d/'hf-train-split.jsonl'
    cmd.ser.model_jsonl_dumpf(gen_from_hf(), out_f)
    logger.success('Wrote: {}', cmd.cwd_rel(out_f))


@app.command()
def write_xl() -> None:
    """
    Writes the XL train split to a local file.
    """
    ctx = cmd.start(
        __file__,
        force=True,
    )
    out_f = ctx.out_d/'xl-train-split.jsonl'
    cmd.ser.model_jsonl_dumpf(gen_xl_from_hf(), out_f)
    logger.success('Wrote: {}', cmd.cwd_rel(out_f))


@app.command()
def make_train_split(
    force: bool = False,
):
    ctx = cmd.start(__file__, force=force)

    dep_f = ctx.out_d.parent/'preprocess_checker_interactor/result.jsonl'
    outf = ctx.out_d/'fresh-train-split.jsonl'

    taken_indices = set(validation_split.gen_indices_from_dep_file())
    taken_indices.update(test_split.gen_indices_from_dep_file())

    wanted_indices = set(
        r['idx']
        for r in tqdm(cmd.ser.jsonl_streamf(dep_f), desc='Collecting wanted row indices')
        if r['type'] == 'diff'
    )

    rows = []
    for r in preprocess_solutions_py.gen_preprocessed_rows():
        if len(r.examples) < 2:
            continue
        if r.idx in taken_indices or r.idx not in wanted_indices:
            continue
        rows.append(r)
    subset = random.sample(rows, 2000)
    subset.sort(key=lambda r: r.idx)
    cmd.ser.model_jsonl_dumpf(subset, outf)
    logger.success('Wrote: {}', cmd.cwd_rel(outf))


if __name__ == '__main__':
    app()
