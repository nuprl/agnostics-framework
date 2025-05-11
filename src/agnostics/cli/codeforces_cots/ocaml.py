import typer

from .common import commands

app = typer.Typer()


PROMPT_OCAML_SUFFIX = '''\

Please reason step by step about your solution approach, \
then provide a complete implementation in OCaml 5. \
Some tips:

Numbers:   + - * / mod   vs.   +. -. *. /. **    (add dots!)
Casts:     float_of_int   int_of_float   int_of_string
Mutation:  refs (:= !) or pass new values recursively
Strings:   split_on_char, String.get => char, use Printf "%c"
Lists:     avoid List.nth; prefer pattern-match / folds / arrays

Put your final answer in a single code block:

```ocaml
<your code here>
```
'''

executor_image_name = 'agnostics-ocaml-executor'
prebuilt_executor_image_name = 'ghcr.io/abgruszecki/agnostics:ml'

command_make_prompts = commands.CmdStandardMakePrompts(__file__, prompt_pl_suffix=PROMPT_OCAML_SUFFIX)
command_generate = commands.CmdGenerate(__file__)
command_verify = commands.CmdVerifyAnswers(executor_image_name=prebuilt_executor_image_name)

for cmd in (command_make_prompts, command_generate, commands.CmdExtractAnswersInstance, command_verify):
    cmd.add_commands(app)


if __name__ == '__main__':
    app()
