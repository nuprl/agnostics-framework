"""
This is based on cpp.py.
"""
import typer

from .common import commands

app = typer.Typer()


PROMPT_RUST_SUFFIX = '''\

Please reason step by step about your solution approach, \
then provide a complete implementation in Rust.

Put your final answer in a single code block:
```rust
<your code here>
```
'''

executor_image_name = 'agnostics-rust-executor'

command_make_prompts = commands.CmdStandardMakePrompts(__file__, prompt_pl_suffix=PROMPT_RUST_SUFFIX)
command_generate = commands.CmdGenerate(__file__)
command_verify = commands.CmdVerifyAnswers(executor_image_name=executor_image_name)

for cmd in (command_make_prompts, command_generate, commands.CmdExtractAnswersInstance, command_verify):
    cmd.add_commands(app)


if __name__ == '__main__':
    app()