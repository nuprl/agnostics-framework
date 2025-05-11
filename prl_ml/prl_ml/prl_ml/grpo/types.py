import abc
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Callable, Union, TypedDict, Literal, Protocol, Any, Tuple, Generator


Role = Union[Literal["user"], Literal["assistant"], Literal["system"]]


class ChatMessage(TypedDict):
    role: Role
    content: str


class Conversation(TypedDict):
    messages: List[ChatMessage]


class RewardFunction(Protocol):
    def __call__(self, items: Iterable[Tuple[str, Dict[str, Any]]]) -> List[float]:
        """
        The type of a reward function. The argument items is a generator of tuples,
        where each tuple contains a completion and a dataset item. The reward function
        should return a list of rewards, one for each item.
        """
        ...


@dataclass
class ModelGenerationData:
    output: str
    row: Dict[str, Any]


class BunchedRewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_reward(self, generation_data: List[ModelGenerationData]) -> List[Tuple[str, List[float]]]:
        """
        Computes all the rewards for the given generation data.

        Returns a list of (name, reward_list) pairs, where name is the name of
        the reward function and reward_list is a list of rewards (one for each item in generation_data).

        Implementation note: for backwards compatibility no more arguments should be added.
        Instead extend `ModelGenerationData`.
        """
        ...

    @staticmethod
    def from_func_list(func_list: List[RewardFunction]) -> 'BunchedRewardFunction':
        return FuncListBunchedRewardFunction(func_list)


class FuncListBunchedRewardFunction(BunchedRewardFunction):
    def __init__(self, func_list: List[RewardFunction]):
        self.func_list = func_list

    def compute_reward(self, generation_data: List[ModelGenerationData]) -> List[Tuple[str, List[float]]]:
        func_input = [
            (item.output, item.row)
            for item in generation_data
        ]
        return [
            (func.__name__, func(func_input)) for func in self.func_list
        ]

class PromptBuilder(Protocol):
    """
    The type of a function that takes a training set item and produces a prompt.

    The prompt is either a string (for a base model) or messages (for a chat
    model). When using a chat model, *do not* apply the chat template yourself.
    The trainer takes care of doing this when appropriate. In particular, we
    also use this function to build prompts for vLLM that does not require us
    to use apply_chat_template.

    The optional completion argument is the response from the LLM. When it
    is specified, the function should produce a prompt with the response.
    In the case of a chat model, the completion is the assistant response.
    """
    def __call__(self, item: Dict[str, Any], completion: Optional[str]) -> Union[str, list[ChatMessage]]:
        ...
