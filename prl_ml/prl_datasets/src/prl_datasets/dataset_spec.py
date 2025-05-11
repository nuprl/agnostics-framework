import json
from pathlib import Path
from typing import Iterable, Iterator, TypeVar, Any, Dict, Callable, Generator
import abc
import csv
from itertools import islice
from lark import Lark, Transformer, v_args

T = TypeVar("T")


class LimitedIterable:
    """
    Wraps another iterable and bounds it by the limit.
    """

    def __init__(self, iterable: Iterable[T], limit: int):
        self.iterable = iterable
        self.limit = limit

    def __iter__(self) -> Iterator[T]:
        yield from islice(self.iterable, self.limit)


class GeneratorFuncIterable:
    """
    An iterable is an object-oriented abomination. What can be done in a
    one-line lambda takes at least 5 lines with an iterable.
    """

    def __init__(self, f: Callable[[], Generator[T, None, None]]):
        self.f = f

    def __iter__(self) -> Iterator[T]:
        yield from self.f()


class DatasetSpec(abc.ABC):
    """
    An abstract class that represents a dataset specification. See below for
    concrete implementations for various representations.
    """

    @classmethod
    def from_string(cls, spec: str) -> "DatasetSpec":
        """
        Parse a dataset specification string and return a DatasetSpec.

        Grammar:
            spec := kind:dataset_name[:flag]*
            kind := csv | jsonl | hub | disk
            flag := key=value | key

        Examples:
            csv:/path/to/file.csv
            jsonl:/path/to/file.jsonl:limit=100
            hub:username/dataset:split=train:limit=1000
            disk:/path/to/dir:limit=500
        """
        return _parser.parse(spec)

    @abc.abstractmethod
    def save(self, items: Iterable[T]) -> None: ...

    @abc.abstractmethod
    def load(self) -> Iterable[T]: ...


class LimitedDatasetSpec(DatasetSpec):
    """
    A wrapper that applies a limit to another DatasetSpec's load() method.
    """

    def __init__(self, inner: DatasetSpec, limit: int):
        self.inner = inner
        self.limit = limit

    def save(self, items: Iterable[T]) -> None:
        self.inner.save(items)

    def load(self) -> Iterable[T]:
        return LimitedIterable(self.inner.load(), self.limit)


_GRAMMAR = """
start: spec

spec: KIND ":" dataset_name (":" flag)*

dataset_name: SEGMENT

flag: key "=" value -> flag_with_value
    | key -> boolean_flag

key: NAME

value: SEGMENT

KIND: "csv" | "jsonl" | "hub" | "disk"
SEGMENT: /[^:]+/
NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
"""


@v_args(inline=True)
class _DatasetSpecTransformer(Transformer):
    """Transforms the parse tree into a DatasetSpec object."""

    def SEGMENT(self, token):
        return token.value

    def NAME(self, token):
        return token.value

    def KIND(self, token):
        return token.value

    def dataset_name(self, segment):
        return str(segment).strip()

    def key(self, name):
        return str(name)

    def value(self, segment):
        return str(segment).strip()

    def flag_with_value(self, key, value):
        if key == "limit":
            value = int(value)
        return (key, value)

    def boolean_flag(self, key):
        return key

    def spec(self, kind, dataset_name, *flags):
        flag_dict = {}
        for flag in flags:
            if isinstance(flag, tuple):
                key, value = flag
                flag_dict[key] = value
            else:
                flag_dict[flag] = True

        limit = flag_dict.pop("limit", None)

        if kind == "csv":
            spec = _CsvFile(Path(dataset_name).resolve(), **flag_dict)
        elif kind == "jsonl":
            spec = _JsonlFile(Path(dataset_name).resolve(), **flag_dict)
        elif kind == "disk":
            spec = _DiskDataset(Path(dataset_name).resolve(), **flag_dict)
        elif kind == "hub":
            if "split" not in flag_dict:
                raise ValueError("split flag is required for hub datasets")
            spec = _HubDataset(dataset_name, **flag_dict)
        else:
            raise ValueError(f"Unknown dataset kind {kind}")

        if limit is not None:
            spec = LimitedDatasetSpec(spec, limit)

        return spec

    def start(self, spec):
        return spec


_parser = Lark(_GRAMMAR, parser="lalr", transformer=_DatasetSpecTransformer())


class _CsvFile(DatasetSpec):
    """
    A dataset specification that reads and writes to a csv file.
    Each call to __iter__ opens the file fresh.
    """

    def __init__(self, path: Path):
        self.path = path

    def save(self, items: Iterable[T]):
        with self.path.open("w") as f:
            writer = csv.writer(f)
            header_written = False
            for item in items:
                if not header_written:
                    writer.writerow(item.keys())
                    header_written = True
                writer.writerow(item.values())

    def load(self) -> Iterable[Dict[str, Any]]:
        def f():
            with self.path.open("r") as f:
                reader = csv.DictReader(f)
                yield from reader

        return GeneratorFuncIterable(f)


class _JsonlFile(DatasetSpec):
    """
    A dataset specification that reads and writes to a jsonl file.
    Each call to __iter__ opens the file fresh.
    """

    def __init__(self, path: Path):
        self.path = path

    def save(self, items: Iterable[T]):
        with self.path.open("w") as f:
            for item in items:
                json.dump(item, f)
                f.write("\n")

    def load(self) -> Iterable[Dict[str, Any]]:
        def f():
            with self.path.open("r") as f:
                for line in f:
                    yield json.loads(line)

        return GeneratorFuncIterable(f)


class _HubDataset(DatasetSpec):
    """
    A dataset specification that reads and writes to a Hugging Face dataset.
    The dataset is loaded once and cached for subsequent iterations.
    """

    def __init__(
        self,
        dataset_name: str,
        push_to_hub: bool = False,
        private: bool = False,
        **flags,
    ):
        self.dataset_name = dataset_name
        self.flags = flags
        self.push_to_hub = push_to_hub
        self.private = private

    def save(self, items: Iterable[T]):
        import datasets

        items_list = list(items)
        if not items_list:
            raise ValueError("Cannot save empty dataset")

        keys = items_list[0].keys()
        data_dict = {key: [item[key] for item in items_list] for key in keys}

        ds = datasets.Dataset.from_dict(data_dict)

        flags = {**self.flags}
        if "name" in flags and flags["name"] is not None:
            flags["config_name"] = flags["name"]
            del flags["name"]

        if self.push_to_hub:
            ds.push_to_hub(repo_id=self.dataset_name, **flags, private=self.private)

    def load(self) -> Iterable[Dict[str, Any]]:
        import datasets

        def f():
            return datasets.load_dataset(self.dataset_name, **self.flags)

        return GeneratorFuncIterable(f)


class _DiskDataset(DatasetSpec):
    """
    A dataset specification that reads and writes to a directory on disk.
    The dataset is loaded once and cached for subsequent iterations.
    """

    def __init__(self, path: Path):
        self.path = path

    def save(self, items: Iterable[T]):
        import datasets

        items_list = list(items)
        if not items_list:
            raise ValueError("Cannot save empty dataset")

        keys = items_list[0].keys()
        data_dict = {key: [item[key] for item in items_list] for key in keys}

        ds = datasets.Dataset.from_dict(data_dict)
        ds.save_to_disk(str(self.path))

    def load(self) -> Iterable[Dict[str, Any]]:
        import datasets

        def f():
            return datasets.Dataset.load_from_disk(str(self.path))

        return GeneratorFuncIterable(f)
