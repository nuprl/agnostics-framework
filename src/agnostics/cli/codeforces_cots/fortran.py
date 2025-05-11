import typer

from .common import commands

app = typer.Typer()


PROMPT_SUFFIX = '''\

Please reason step by step about your solution approach, \
then provide a complete implementation in Fortran 90. \
Some tips:

Always begin each scope with implicit none, pick explicit kinds via \
selected_*_kind, and declare proper lengths-character(len=*) is legal only for \
dummy arguments, not locals.  Strings are blank-padded: call len_trim before \
iterating, and store dynamic text in deferred-length allocatables \
(character(len=:), allocatable :: s).  List-directed read(*,*) arr does not \
auto-size arrays; read a count first, then allocate and read, or tokenize a \
line manually.  When translating 0-based formulas (heaps, bit positions) \
remember Fortran arrays default to 1-based; if you want 0-based, declare lower \
bounds.  Use real literals (2.0d0, 1.0_rk) to avoid silent integer division, \
and guard against overflow when exponentiating integers.  For frequency tables, \
allocate an array or use findloc; Fortran lacks native dicts/sets, so you must \
implement search yourself.  Prefer array intrinsics (sum, count, pack) over \
hand-rolled loops, and keep helper procedures inside a contains section or \
module so interfaces are explicit.  return inside the main program is \
non-idiomatic; use structured blocks or stop.  Never print interactive prompts \
in batch solutions; just read, compute, and write.

Put your final answer in a single code block:

```fortran
<your code here>
```
'''

executor_image_name = 'agnostics-fortran-executor'
prebuilt_executor_image_name = 'ghcr.io/abgruszecki/agnostics:f90'

command_make_prompts = commands.CmdStandardMakePrompts(__file__, prompt_pl_suffix=PROMPT_SUFFIX)
command_generate = commands.CmdGenerate(__file__)
command_verify = commands.CmdVerifyAnswers(executor_image_name=prebuilt_executor_image_name)

for cmd in (command_make_prompts, command_generate, commands.CmdExtractAnswersInstance, command_verify):
    cmd.add_commands(app)


if __name__ == '__main__':
    app()
