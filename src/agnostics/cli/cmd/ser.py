"""
`ser` is short for "serialization".

This module provides common de/serialization utilities.
"""
import json as json
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Protocol, TextIO

from loguru import logger


type Pathlike = str | bytes | PathLike[str] | PathLike[bytes]


class ModelClassProto[A](Protocol):
    def model_validate(self, obj: Any) -> A: ...


class ModelInstanceProto(Protocol):
    def model_dump(self, *, mode: str | Literal['json', 'python'] = 'json') -> dict: ...


def json_dumpf(obj, pathlike: Pathlike):
    with open(pathlike, 'w') as fh:
        json.dump(obj, fh)


def json_loadf(pathlike: Pathlike):
    with open(pathlike, 'r') as fh:
        return json.load(fh)


def jsonl_streamf(pathlike: Pathlike, start: int = 0, end: int = -1, allow_malformed_lines: bool = False):
    with open(pathlike, 'r') as fh:
        for i, line in enumerate(fh):
            if i < start:
                continue
            if end != -1 and i >= end:
                break
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if allow_malformed_lines:
                    logger.warning('Malformed line: {}:{}', pathlike, i+1)
                else:
                    raise


def jsonl_loadf(pathlike: Pathlike):
    return list(jsonl_streamf(pathlike))


def jsonl_dumpf(data: Iterable[Any], pathlike: Pathlike):
    with open(pathlike, 'w') as fh:
        jsonl_dumpfh(data, fh)


def jsonl_dumpfh(data: Iterable[Any], fh: TextIO):
    for row in data:
        print(json.dumps(row), file=fh)


def model_jsonl_streamf[A](
    model: ModelClassProto[A],
    pathlike: Pathlike,
    start: int = 0,
    end: int = -1,
) -> Iterator[A]:
    for r in jsonl_streamf(pathlike, start, end):
        yield model.model_validate(r)


def model_jsonl_loadf[A](
    model: ModelClassProto[A],
    pathlike: Pathlike,
) -> list[A]:
    return list(model_jsonl_streamf(model, pathlike))


def model_jsonl_dumpf(
    data: Iterable[ModelInstanceProto],
    pathlike: Pathlike,
):
    jsonl_dumpf(
        (r.model_dump(mode='json') for r in data),
        pathlike,
    )

def str_dumpf(obj: Any, pathlike: str | Path):
    Path(pathlike).write_text(str(obj))