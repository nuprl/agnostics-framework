"""
The purpose of this module is to make it easy to change the learning rate
scheduler in an experiment. For example, you can write code such as this:


```python
sched_str = "cosine(warmup_ratio=0.1)"
sched_ctor = get_scheduler(sched_str)
scheduler = sched_ctor(optimizer, num_training_steps)
```

Notice that `sched_ctor` takes in both the optimizer and the number of training
steps. These are conveniences, since the number of steps is typically
computed or specified.

What follows is a simplified syntax for scheduler strings. Notice that the
parameters are not comma separated.

```
scheduler ::= scheduler_name "(" param* ")"

scheduler_name ::= "constant" | "cosine" | "linear"

param ::= param_name "=" param_value

param_name ::= "num_warmup_steps" | "num_training_steps" | "warmup_ratio"

param_value ::= number | "(" number "," number ")"
```
"""

from lark import Lark, Transformer
from transformers import get_scheduler as transformers_get_scheduler
from typing import Callable
from torch.optim.optimizer import Optimizer

__all__ = ["get_scheduler"]

_scheduler_grammar = """
    scheduler: constant | cosine | linear

    constant: "constant"
    cosine: "cosine" "(" cosine_args ")"
    linear: "linear" "(" linear_args ")"

    cosine_args: cosine_param*
    linear_args: linear_param*

    cosine_param: num_warmup_steps | warmup_ratio
    linear_param: num_warmup_steps | warmup_ratio

    num_warmup_steps: "num_warmup_steps" "=" POS_INT
    warmup_ratio: "warmup_ratio" "=" POS_FLOAT
    POS_INT: /[+]?[0-9]+/
    POS_FLOAT: /[+]?[0-9]*\\.?[0-9]+([eE][+-]?[0-9]+)?/

    %import common.WS
    %ignore WS
"""


def _assert_no_duplicates(items):
    """Assert that there are no duplicate parameters in the scheduler string."""
    seen = set()
    for key, _ in items:
        if key in seen:
            raise ValueError(f"Duplicate parameter: {key}")
        seen.add(key)

def _process_params(params, num_training_steps):
    """
    We add a warmup ratio argument to allow us the trainer to compute the warmup
    steps.
    """
    if "warmup_ratio" in params:
        params["num_warmup_steps"] = int(num_training_steps * params["warmup_ratio"])
        del params["warmup_ratio"]

    return params

class _SchedulerTransformer(Transformer):
    """Parses a scheduler string into a scheduler factory."""

    def number(self, n):
        return float(n[0])

    def num_warmup_steps(self, items):
        return ("num_warmup_steps", int(float(items[0])))
    
    def warmup_ratio(self, items):
        return ("warmup_ratio", float(items[0]))

    def cosine_param(self, items):
        return items[0]

    def linear_param(self, items):
        return items[0]

    def cosine_args(self, items):
        _assert_no_duplicates(items)
        return dict(items)

    def linear_args(self, items):
        _assert_no_duplicates(items)
        return dict(items)

    def constant(self, _):
        return lambda optimizer, num_training_steps: transformers_get_scheduler(
            "constant", optimizer=optimizer
        )

    def cosine(self, items):
        params = items[0]
        return lambda optimizer, num_training_steps: transformers_get_scheduler(
            "cosine", optimizer=optimizer, num_training_steps=num_training_steps, **_process_params(params, num_training_steps)
        )

    def linear(self, items):
        params = items[0]
        return lambda optimizer, num_training_steps: transformers_get_scheduler(
            "linear", optimizer=optimizer, num_training_steps=num_training_steps, **_process_params(params, num_training_steps)
        )

    def scheduler(self, items):
        return items[0]


_parser = Lark(
    _scheduler_grammar,
    start="scheduler",
    parser="lalr",
    transformer=_SchedulerTransformer(),
)


def get_scheduler(scheduler_str: str) -> Callable[[Optimizer], object]:
    """
    Create a scheduler factory function from a string.

    Args:
        scheduler_str: String specification of the scheduler

    Returns:
        A callable that creates a scheduler instance when called with an optimizer

    Examples:
        >>> scheduler = get_scheduler("cosine(num_warmup_steps=1000, num_training_steps=10000)")
        >>> scheduler(optimizer)
    """
    try:
        return _parser.parse(scheduler_str)
    except Exception as e:
        raise ValueError(f"Invalid scheduler string: {e}")
