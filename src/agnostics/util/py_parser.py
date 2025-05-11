import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree


py_language = Language(tspython.language())
py_parser = Parser(py_language)


def check_is_python(
    s: str,
    old_tree: Tree | None = None,
) -> tuple[bool, Tree]:

    tree = py_parser.parse(s.encode())
    return not tree.root_node.has_error, tree