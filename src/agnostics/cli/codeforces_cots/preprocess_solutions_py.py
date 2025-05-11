"""
Processes the solutions_py part of the dataset:
"""
from pathlib import Path
import random
from typing import Iterable, Iterator

from loguru import logger
from tqdm import tqdm
import typer

from agnostics.schema.solutions_py import SolutionsRow, SolutionsRowWithAnswer, IOExample
from .. import cmd

app = typer.Typer()


def gen_output_rows(input_gen, quiet: bool = False):
    for idx, in_row in enumerate(input_gen):
        if in_row['description'] is None:
            continue

        structured_examples: list[IOExample] = []
        for example in in_row.get('examples') or []:
            try:
                structured_examples.append(IOExample(
                    input=example['input'],
                    output=example['output'],
                ))
            except IndexError:
                if not quiet:
                    logger.warning('Skipping example (at row {}): {}', idx, example)
                continue

        if not structured_examples:
            if not quiet:
                logger.warning('No examples, skipping row {}', idx)
            continue

        response: str = in_row['generation']

        r = SolutionsRow(
            idx=idx,
            source_id=in_row['id'],
            prompt=in_row['prompt'],
            response=response,

            problem_statement=in_row['description'],
            time_limit=in_row['time_limit'],
            memory_limit=in_row['memory_limit'],
            input_format=in_row['input_format'],
            output_format=in_row['output_format'],
            examples=structured_examples,
            problem_notes=in_row['note'],

            title=in_row['title'],
            contest_name=in_row['contest_name'],
            contest_start_year=in_row['contest_start_year'],
        )
        yield r


def gen_preprocessed_rows() -> Iterator[SolutionsRow]:
    """
    This function can be imported by other modules.
    """
    import datasets
    ds = datasets.load_dataset('open-r1/codeforces-cots', 'solutions_py_decontaminated', split='train')

    yield from gen_output_rows(tqdm(ds, desc='Processing solutions_py_decontaminated'), quiet=True)


def gen_preprocessed_rows_with_answers() -> Iterator[SolutionsRowWithAnswer]:
    from agnostics.util.code_finder import thinks_end_rx, find_final_answer_block
    for row in gen_preprocessed_rows():
        m = thinks_end_rx.search(row.response)
        if m is None:
            yield SolutionsRowWithAnswer(
                **row.model_dump(),
                answer=None,
            )
            continue
        offset = m.end()
        answer = find_final_answer_block(row.response, offset)
        yield SolutionsRowWithAnswer(
            **row.model_dump(),
            answer=answer,
        )


@app.command()
def main():
    """
    Outputs all the preprocessed rows to a file.
    """
    ctx = cmd.start(__file__, force=True)
    outf = ctx.out_d / 'result.jsonl'

    cmd.ser.model_jsonl_dumpf(gen_preprocessed_rows(), outf)
    logger.success('Wrote: {}', cmd.cwd_rel(outf))


if __name__ == '__main__':
    app()
