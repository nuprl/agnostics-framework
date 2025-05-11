"""
The purpose of this module is to make it easy to change the optimizer you are
using in an experiment. For example, you can write code such as this:

```python
from prl_ml.optimizer import get_optimizer

opt_string = "Adam(lr=0.001 betas=(0.9, 0.999)" 
opt_ctor = get_optimizer(opt_string)
optimizer = opt_ctor(model.parameters())
```

Of course, `opt_string` can be a command-line argument, logged to Wandb or
Tensorboard, and so on. What follows is a simplified syntax for optimizer
strings. Notice that the parameters are not comma separated.

```
optimizer ::= optimizer_name "(" param* ")"

optimizer_name ::= "Adam" | "AdamW"

param ::= param_name "=" param_value

param_name ::= "lr" | "betas" | "eps" | "weight_decay" | "amsgrad" | "fused"

param_value ::= number | "(" number "," number ")"
```
"""
from lark import Lark, Transformer
import torch.optim as optim
from torch.optim.optimizer import ParamsT
from typing import Callable

__all__ = ["get_optimizer"]


_optimizer_grammar = """
    optimizer: adam | adamw

    adam: "Adam" "(" adam_args ")"
    adamw: "AdamW" "(" adamw_args ")"

    adam_args: adam_param*
    adamw_args: adamw_param*

    adam_param: lr | betas | eps | weight_decay | amsgrad | fused
    adamw_param: lr | betas | eps | weight_decay | amsgrad | fused

    lr: "lr" "=" NUMBER
    betas: "betas" "=" "(" NUMBER "," NUMBER ")"
    eps: "eps" "=" NUMBER
    weight_decay: "weight_decay" "=" NUMBER
    amsgrad: "amsgrad" "=" BOOL
    fused: "fused" "=" BOOL

    BOOL: "True" | "False"
    NUMBER: /[+-]?[0-9]*\\.?[0-9]+([eE][+-]?[0-9]+)?/

    %import common.WS
    %ignore WS
    COMMA: ","
"""


def _assert_no_duplicates(items):
    """
    Assert that there are no duplicate parameters in the optimizer string.
    """
    seen = set()
    for key, _ in items:
        if key in seen:
            raise ValueError(f"Duplicate parameter: {key}")
        seen.add(key)


class _OptimizerTransformer(Transformer):
    """
    Parses an optimizer string into an optimizer instance.
    """

    def number(self, n):
        return float(n[0])

    def bool(self, b):
        return b[0] == "True"

    def lr(self, items):
        return ("lr", float(items[0]))

    def betas(self, items):
        return ("betas", (float(items[0]), float(items[1])))

    def eps(self, items):
        return ("eps", float(items[0]))

    def weight_decay(self, items):
        return ("weight_decay", float(items[0]))

    def amsgrad(self, items):
        return ("amsgrad", items[0] == "True")

    def fused(self, items):
        return ("fused", items[0] == "True")

    def adam_param(self, items):
        return items[0]

    def adamw_param(self, items):
        return items[0]

    def adam_args(self, items):
        params = dict(items)
        _assert_no_duplicates(items)
        return params

    def adamw_args(self, items):
        params = dict(items)
        _assert_no_duplicates(items)
        return params

    def adam(self, items):
        return lambda params: optim.Adam(params, **items[0])

    def adamw(self, items):
        return lambda params: optim.AdamW(params, **items[0])

    def optimizer(self, items):
        return items[0]


_parser = Lark(
    _optimizer_grammar,
    start="optimizer",
    parser="lalr",
    transformer=_OptimizerTransformer(),
)


def get_optimizer(optimizer_str: str) -> Callable[[ParamsT], optim.Optimizer]:
    """
    Create an optimizer factory function from a string.

    Args:
        optimizer_str: String specification of the optimizer

    Returns:
        A callable that creates an optimizer instance when called with parameters
    """
    try:
        return _parser.parse(optimizer_str)
    except Exception as e:
        raise ValueError(f"Invalid optimizer string: {e}")
