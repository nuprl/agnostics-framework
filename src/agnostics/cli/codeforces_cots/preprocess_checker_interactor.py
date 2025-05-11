import json
import os
from pathlib import Path
from typing import Any, Iterator

from loguru import logger
import typer

from agnostics.util.code_finder import find_final_answer_block, thinks_end_rx
from .. import cmd

app = typer.Typer()


def gen_output_rows():
    import datasets
    ds: Any = datasets.load_dataset('open-r1/codeforces-cots', 'solutions_py_decontaminated', split='train')
    wanted_ids_list = []
    wanted_ids_set = set()
    for r in ds:
        wanted_ids_list.append(r['id'])
        wanted_ids_set.add(r['id'])

    checker_ds: Any = datasets.load_dataset('open-r1/codeforces-cots', 'checker_interactor', split='train')
    checker_dict = {
        r['id']: r['generation'] for r in checker_ds if r['id'] in wanted_ids_set
    }

    for row_idx, row_id in enumerate(wanted_ids_list):
        r = {'idx': row_idx, 'id': row_id }

        offset = 0
        if m := thinks_end_rx.search(checker_dict[row_id]):
            offset = m.end()
        else:
            logger.warning('No thinks end at: {}', row_idx)

        json_str = find_final_answer_block(
            checker_dict[row_id],
            offset=offset,
            answer_must_be_valid_python=False,
        )
        if json_str is None:
            logger.warning('No JSON found at: {}', row_idx)
            continue

        try:
            json_obj = json.loads(json_str)
        except Exception as e:
            logger.warning('Error parsing JSON at {}: {}', row_idx, e)
            continue

        r.update(json_obj)
        yield r


def gen_from_hf() -> Iterator[dict[str, Any]]:
    import datasets
    ds: Any = datasets.load_dataset('nuprl/agnostics-codeforces-cots-checker-interactor', split='test')
    for r in ds:
        yield r


@app.command()
def write(
    force: bool = False,
):
    ctx = cmd.start(__file__, force=force)
    outf = ctx.out_d / 'result.jsonl'

    cmd.ser.jsonl_dumpf(gen_output_rows(), outf)
    logger.success('Wrote: {}', cmd.cwd_rel(outf))


@app.command()
def push_to_hf():
    ctx = cmd.start(__file__, readonly=True)

    dep_f = ctx.out_d / 'result.jsonl'
    if not dep_f.exists():
        logger.error('Missing input file: {}', dep_f)
        raise typer.Abort()

    import datasets
    import huggingface_hub as hfh

    tok = os.getenv('HF_TOKEN')
    if not tok:
        logger.error('Missing environment variable: HF_TOKEN')

    ds: Any = datasets.load_dataset('json', data_files={
        'test': str(dep_f),
    })

    ds.push_to_hub('nuprl-staging/agnostics-codeforces-cots-checker-interactor')


if __name__ == '__main__':
    app()
