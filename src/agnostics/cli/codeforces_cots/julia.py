"""
This is based on lua.py.
"""
import typer

from .common import commands

app = typer.Typer()


PROMPT_JULIA_SUFFIX = '''\

Please reason step by step about your solution approach, \
then provide a complete implementation in Julia 1.11.

Put your final answer in a single code block:
```julia
<your code here>
```
'''

executor_image_name = 'agnostics-julia-executor'
prebuilt_executor_image_name = 'ghcr.io/abgruszecki/agnostics:jl'

command_make_prompts = commands.CmdStandardMakePrompts(__file__, prompt_pl_suffix=PROMPT_JULIA_SUFFIX)
command_generate = commands.CmdGenerate(__file__)
command_verify = commands.CmdVerifyAnswers(executor_image_name=prebuilt_executor_image_name)

for cmd in (command_make_prompts, command_generate, commands.CmdExtractAnswersInstance, command_verify):
    cmd.add_commands(app)


if __name__ == '__main__':
    app()