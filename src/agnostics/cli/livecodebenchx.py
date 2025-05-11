import typer
from loguru import logger

from . import cmd

app = typer.Typer()


def make_prompt_from_lcbx_row(row: dict, lang: str) -> str:
    return f"""\
# Problem
{row['question_content']}

# Task
Provide a full implementation of the specified program in a Markdown code block.
Use the following programming language: {lang}
"""


@app.callback()
def empty_callback():
    pass


@app.command()
def write_prompts(
    lang: str,
):
    ctx = cmd.start(__file__, sub_out_d=lang)

    out_f = ctx.out_d/'result.jsonl'

    import datasets

    ds = datasets.load_dataset('nuprl/LiveCodeBench-X', split='test')

    out_row_gen = (
        {
            'idx': row['question_id'],
            'prompt': make_prompt_from_lcbx_row(row, lang=lang),
        }
        for row in ds
    )

    cmd.ser.jsonl_dumpf(out_row_gen, out_f)
    logger.success('Wrote: {}', cmd.cwd_rel(out_f))


if __name__ == '__main__':
    app()