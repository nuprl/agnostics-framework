import random
from typing import Any, Iterator

from tqdm import tqdm
import typer
from loguru import logger

from . import preprocess_solutions_py
from .. import cmd

app = typer.Typer()


def make_solutions_row(dictlike: Any) -> preprocess_solutions_py.SolutionsRow:
    return preprocess_solutions_py.SolutionsRow(
        idx=dictlike['idx'],
        source_id=dictlike['source_id'],
        prompt=dictlike['prompt'],
        response=dictlike['response'],
        problem_statement=dictlike['problem_statement'],
        time_limit=dictlike['time_limit'],
        memory_limit=dictlike['memory_limit'],
        input_format=dictlike['input_format'],
        output_format=dictlike['output_format'],
        examples=dictlike['examples'],
        problem_notes=dictlike['problem_notes'],
        title=dictlike['title'],
        contest_name=dictlike['contest_name'],
        contest_start_year=dictlike['contest_start_year'],
    )


def gen_from_hf() -> Iterator[preprocess_solutions_py.SolutionsRow]:
    """
    A helper function for other modules: loads the test split from the HF dataset and yields the rows.
    """
    import datasets
    ds = datasets.load_dataset('nuprl/agnostics-codeforces-cots', split='test')

    for in_row in tqdm(ds, desc='Processing the test split'):
        yield make_solutions_row(in_row)


def gen_from_dep_file(
    start: int = 0,
    end: int = -1,
) -> Iterator[preprocess_solutions_py.SolutionsRow]:
    ctx = cmd.start(__file__, readonly=True)
    dep_f = ctx.out_d/'fresh-test-split.jsonl'
    if not dep_f.exists():
        logger.error('Dependency file does not exist: {}', dep_f)
        raise typer.Exit(1)

    yield from cmd.ser.model_jsonl_streamf(preprocess_solutions_py.SolutionsRow, dep_f, start, end)


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
    out_f = ctx.out_d/'hf-test-split.jsonl'
    cmd.ser.model_jsonl_dumpf(gen_from_hf(), out_f)
    logger.success('Wrote: {}', cmd.cwd_rel(out_f))


@app.command()
def make_test_split(
    force: bool = False,
):
    """
    Outputs 5 manually picked and 100 random preprocessed codeforces-cots rows
    which allow a diff-based answer checker.
    """
    ctx = cmd.start(__file__, force=force)

    dep_f = ctx.out_d/'../preprocess_checker_interactor/result.jsonl'
    outf = ctx.out_d/'fresh-test-split.jsonl'

    wanted_indices = set(
        r['idx']
        for r in tqdm(cmd.ser.jsonl_streamf(dep_f), desc='Collecting wanted row indices')
        if r['type'] == 'diff'
    )

    special_indices = (0, 1, 2, 4, 6)
    rows = []
    subset = []
    for r in preprocess_solutions_py.gen_preprocessed_rows():
        if len(r.examples) < 2:
            continue
        if r.idx not in wanted_indices:
            continue
        if r.idx in special_indices:
            subset.append(r)
        else:
            rows.append(r)
    assert len(subset) == len(special_indices), f'{len(subset)=} != {len(special_indices)=}'
    subset.extend(random.sample(rows, 100))
    subset.sort(key=lambda r: r.idx)
    cmd.ser.model_jsonl_dumpf(subset, outf)
    logger.success('Wrote: {}', cmd.cwd_rel(outf))


if __name__ == '__main__':
    app()
