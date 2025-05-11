import typer

from .common import commands

app = typer.Typer()


PROMPT_R_SUFFIX = '''\

Please reason step by step about your solution approach, \
then provide a complete implementation in R version 4.

Use `readLines(con = file("stdin"))` to read input from stdin. Optionally, use the `n` argument to read the first `n` lines.
For example:
```r
input <- readLines(con = file("stdin"), n = 1)
n <- as.integer(input)
cat(n) # print the first line of input
```
Also, use `cat` to print output to stdout. For example:
```r
cat(n)
```
Please do not use `print` to print output.

Put your final answer in a single code block:
```r
<your code here>
```
'''

executor_image_name = 'agnostics-r-executor'
prebuilt_executor_image_name = 'ghcr.io/abgruszecki/agnostics:r'

command_make_prompts = commands.CmdStandardMakePrompts(__file__, prompt_pl_suffix=PROMPT_R_SUFFIX)
command_generate = commands.CmdGenerate(__file__)
command_verify = commands.CmdVerifyAnswers(executor_image_name=prebuilt_executor_image_name)

for cmd in (command_make_prompts, command_generate, commands.CmdExtractAnswersInstance, command_verify):
    cmd.add_commands(app)


if __name__ == '__main__':
    app()
