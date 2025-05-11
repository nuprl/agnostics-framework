import re
from typing import List, Dict, Any, Optional, Generator, Tuple
from argparse import ArgumentParser
import os
from .ray_trainer import train


def prompt_builder(item: Dict[str, Any], completion: Optional[str]) -> str:
    parts = [
        "Reason step-by-step through the following problem before giving the answer.\n",
        item["question"],
    ]
    if completion:
        parts.extend(["", completion])
    return "\n".join(parts)

def extract_numbers(text: str) -> List[float]:
    """
    Extracts all numbers from a string.
    """
    return [float(num) for num in re.findall(r'-?\d*\.?\d+', text)]

def result_on_last_line(items: Generator[Tuple[str, Dict[str, Any]], None, None]) -> List[float]:
    """
    A very naive reward function that may be wrong: does the answer appear anywhere on the last line of output?
    """
    rewards = [ ] 
    for output, item in items:
        final_line = output.strip().split("\n")[-1]
        numbers = extract_numbers(final_line)
        if len(numbers) == 0:
            rewards.append(0.0)
        elif len(numbers) > 1:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if numbers[0] == item["answer"] else 0.0)
    return rewards

def main():
    if "HF_HOME" not in os.environ:
        print(
            "HF_HOME is not set: will download models and data to your home directory."
        )
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument(
        "--train-dataset-spec",
        type=str,
        default="hub:nuprl/engineering-llm-systems:name=gsm8k:split=train",
    )
    parser.add_argument(
        "--test-dataset-spec",
        type=str,
        default="hub:nuprl/engineering-llm-systems:name=math_word_problems:split=test",
    )
    args = parser.parse_args()

    train(
        reward_funcs=[result_on_last_line],
        prompt_builder=prompt_builder,
        stop=None,
        **vars(args)
    )


if __name__ == "__main__":
    main()
