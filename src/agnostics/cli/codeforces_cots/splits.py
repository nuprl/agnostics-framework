import os
from typing import Any

import typer
from loguru import logger

from .. import cmd

app = typer.Typer()


@app.callback()
def empty_callback():
    pass


@app.command()
def push_splits():
    """
    Pushes all the splits to the HF Hub. Reads the HF token from the HF_TOKEN env var.
    """
    import datasets
    import huggingface_hub as hfh

    ctx = cmd.start(__file__, readonly=True)

    dep_train_f = ctx.out_d.parent/'train_split/fresh-train-split.jsonl'
    dep_validation_f = ctx.out_d.parent/'validation_split/fresh-validation-split.jsonl'
    dep_test_f = ctx.out_d.parent/'test_split/fresh-test-split.jsonl'

    tok = os.getenv('HF_TOKEN')
    if not tok:
        logger.error('Missing environment variable: HF_TOKEN')

    for dep_f in (dep_train_f, dep_validation_f, dep_test_f):
        if not dep_f.exists():
            logger.error('Missing input file: {}', dep_f)

    hfh.login(tok)

    ds: Any = datasets.load_dataset('json', data_files={
        'train': str(dep_train_f),
        'validation': str(dep_validation_f),
        'test': str(dep_test_f),
    })

    ds.push_to_hub('nuprl-staging/agnostics-codeforces-cots')


if __name__ == '__main__':
    app()
