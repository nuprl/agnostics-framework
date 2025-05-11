#!/usr/bin/env -S uv run
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


@app.command()
def write_pass1_tsv(
    input_files: Annotated[list[Path], typer.Argument(
        help="JSONL files with Agnostics verifier results (typically */verify.jsonl)",
    )],
):
    """
    Compute pass1 scores for Agnostics verifier results stored as JSONL.
    """
    import pandas as pd # intentional delayed import

    print('file\tsuccess-rate')
    for input_file in input_files:
        df = pd.read_json(input_file, lines=True)
        score = (df['status'] == 'success').mean()
        print(f'{input_file}\t{score:.2%}')


if __name__ == '__main__':
    app()
