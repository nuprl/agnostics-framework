"""
A module with (1) a "start" function which CLI commands should have in common,
and (2) common utilities.
"""
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast
import sys

from loguru import logger
import typer

from . import ser as ser


class CmdCtx():
    resources_d: Path
    root_out_d: Path
    out_d: Path

    def __init__(
        self,
        resources_d: Path,
        root_out_d: Path,
        out_d: Path,
    ):
        self.resources_d = resources_d
        self.root_out_d = root_out_d
        self.out_d = out_d

    def to_kwargs(self) -> dict[str, Any]:
        return {
            'resources_d': self.resources_d,
            'root_out_d': self.root_out_d,
            'out_d': self.out_d,
        }


def start_no_ctx(
    out_d: Path | None,
    store_logs: bool = True,
):

    def patcher(r):
        def _short_level():
            l = r['level'].name
            match l:
                case 'TRACE':
                    return 'TRCE'
                case 'DEBUG':
                    return 'DBG'
                case 'INFO':
                    return 'INFO'
                case 'SUCCESS':
                    return 'SCCS'
                case 'WARNING':
                    return 'WARN'
                case 'ERROR':
                    return 'ERR'
                case 'CRITICAL':
                    return 'CRIT'
                case other:
                    return other

        def _better_module():
            match r['module']:
                case '__init__':
                    return r['name'].rsplit('.', 1)[-1]
                case other:
                    if other and other[0] != '_':
                        return other
                    else:
                        return r['name']

        r['extra']['short-level'] = _short_level()
        r['extra']['better-module'] = _better_module()

    short_loguru_fmt = (
        '<green>{time:YYYYMMDDTHHmmss}</green>'
        ' <level>{extra[short-level]: <4}</level>'
        ' {extra[better-module]}:{line}'
        ' - {message}'
    )

    longer_short_loguru_fmt = (
        '<green>{time:YYYYMMDDTHHmmss}</green>'
        ' <level>{extra[short-level]: <4}</level>'
        ' {name}:{line}'
        ' - {message}'
    )

    handlers = [
        dict(sink=sys.stderr, format=short_loguru_fmt),
    ]
    if out_d and store_logs:
        handlers.append(dict(sink=out_d/'log.txt', format=longer_short_loguru_fmt))

    logger.configure(
        handlers=handlers,
        patcher=patcher,
    )


def start(
    file_attr,
    *,
    out_d: Path | None = None,
    sub_out_d: str | Path | None = None,
    store_logs: bool = True,
    readonly: bool = False,
    force: bool = False,
) -> CmdCtx:
    resources_d = Path(__file__).parent/'resources'/Path(__file__).stem
    root_outd = Path.cwd()/'out'

    cli_package_dir = Path(__file__).parent.parent
    step_relpath = Path(file_attr).absolute().relative_to(cli_package_dir)
    if out_d is None:
        out_d = root_outd/step_relpath.with_suffix('')
    if sub_out_d is not None:
        out_d = out_d/sub_out_d
    if readonly:
        start_no_ctx(None)
    else:
        if not force and out_d.exists() and any(out_d.iterdir()):
            start_no_ctx(None)
            logger.error('Output dir already exists: {}', cwd_rel(out_d))
            logger.error('Use --force to overwrite an existing output dir.')
            raise typer.Exit(1)
        out_d.mkdir(parents=True, exist_ok=True)

        start_no_ctx(out_d, store_logs=store_logs)

    return CmdCtx(
        resources_d=resources_d,
        root_out_d=root_outd,
        out_d=out_d,
    )


def cwd_rel(p: Path) -> Path:
    return p.absolute().relative_to(Path.cwd(), walk_up=True)


type Jsonable = str|int|bool|None|list['Jsonable']|dict[str, 'Jsonable']
type JsonableDict = dict[str, Jsonable]


def typecheck_jsonable(d: JsonableDict) -> dict:
    """
    A helper: check via types that a dict can be converted to a JSON object.
    """
    return cast(dict, d)


def InputDirArg(decl: str | None = None):
    typer_decls = [decl] if decl else []
    return typer.Argument(..., *typer_decls, exists=True, file_okay=False, dir_okay=True, readable=True)


def InputDirOption(decl: str | None = None):
    typer_decls = [decl] if decl else []
    return typer.Option(..., *typer_decls, exists=True, file_okay=False, dir_okay=True, readable=True)


def InputFileOption(decl: str | None = None):
    typer_decls = [decl] if decl else []
    return typer.Option(..., *typer_decls, exists=True, file_okay=True, dir_okay=False, readable=True)

