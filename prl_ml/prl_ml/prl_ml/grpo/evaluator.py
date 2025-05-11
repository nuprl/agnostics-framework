from typing import List, Any, assert_never
import time
import json
import ray
import torch
from prl_ml.datasets.dataset_spec import DatasetSpec
from .types import BunchedRewardFunction, ModelGenerationData


@ray.remote
class Evaluator:
    """
    We use an actor for evaluation so that evaluation can run concurrently with
    training. The actor uses an OutputGenerator, which is likely the same
    output generator that the trainer is using to sample group outputs. Since
    actors are single-threaded, as long as evaluation starts before the
    trainer tries to update the model weights on the OutputGenerator, we will
    end up evaluating the right object. i.e., use this pattern in the training
    loop:

    ```python
    if ... evaluation conditions are met ...:
        eval_future = evaluator.evaluate.remote(step) # No ray.get

    ... training code that updates the trainer's copy of the model ...

    if eval_future is not None:
        ray.get(eval_future)
        eval_future = None

    ray.get(object_generator.update_model.remote(...))
    ```

    Since both the trainer and the evaluator use the same OutputGenerator,
    and training cannot proceed without group outputs, you must ensure you
    have the current group outputs to ensure that evaluation doesn't block
    group output generation. Use this pattern:

    ```python
    group_outputs = ray.get(output_generator.generate.remote(...))
    eval_future = evaluator.evaluate.remote(step)

    ... training code that uses group_outputs ...
    ```
    """

    def __init__(
        self,
        output_generator: ray.ObjectRef,
        test_dataset_spec: str | dict[str, str],
        reward_funcs: BunchedRewardFunction | dict[str, BunchedRewardFunction],
        logger: ray.ObjectRef,
    ):
        self._output_generator = output_generator
        self._reward_funcs = reward_funcs
        self._many_reward_funcs = isinstance(reward_funcs, dict)
        self._logger = logger
        if isinstance(test_dataset_spec, str):
            assert not self._many_reward_funcs
            self._many_datasets = False
            test_data = DatasetSpec.from_string(test_dataset_spec).load()
        elif isinstance(test_dataset_spec, dict):
            self._many_datasets = True
            test_data = {
                k: DatasetSpec.from_string(v).load() for k, v in test_dataset_spec.items()
            }
        else:
            assert_never(test_dataset_spec)
        self._test_data = test_data

    def evaluate(self, step: int):
        start_time = time.perf_counter()

        if not self._many_datasets:
            completions = ray.get(
                self._output_generator.generate.remote(self._test_data, 0.0)
            )
            agg_rewards = self._evaluate_single_dataset(
                step, self._test_data, self._reward_funcs, completions, ""
            )
            mean_reward = agg_rewards.mean().item()
        else:
            iter_ = iter(self._test_data.items())
            mean_rewards_list = []


            completions_future = None
            elt = next(iter_, None)
            if elt:
                completions_future = self._output_generator.generate.remote(elt[1], 0.0)

            def process_elt():
                nonlocal elt, completions_future, mean_rewards_list

                if not elt:
                    return
                dataset_name, cur_dataset = elt
                reward_func = self._get_reward_func(dataset_name)
                agg_rewards = self._evaluate_single_dataset(
                    step,
                    cur_dataset,
                    reward_func,
                    ray.get(completions_future),
                    dataset_name
                )
                mean_rewards_list.append(agg_rewards.mean().item())

            while next_elt := next(iter_, None):
                next_completions_future = self._output_generator.generate.remote(next_elt[1], 0.0)
                process_elt()
                elt = next_elt
                completions_future = next_completions_future

            process_elt()
            mean_reward = sum(mean_rewards_list) / len(mean_rewards_list)

        self._logger.add_scalar.remote(
            "test/mean_reward", mean_reward, step
        )
        self._logger.add_scalar.remote(
            "timer/eval_time", time.perf_counter() - start_time, step
        )

    def _evaluate_single_dataset(
        self,
        step: int,
        dataset: Any,
        reward_func: BunchedRewardFunction,
        completions: List[str],
        extra_logger_key: str,
    ):
        logger = self._logger
        extra_reward_key = extra_table_key = ""
        if extra_logger_key:
            extra_reward_key = extra_logger_key + "/"
            extra_table_key = "/" + extra_logger_key

        reward_func_args = [
            ModelGenerationData(c, dataset[i]) for i, c in enumerate(completions)
        ]

        transposed_reward_names = []
        transposed_rewards = []
        for name, reward in reward_func.compute_reward(reward_func_args):
            transposed_reward_names.append(name)
            transposed_rewards.append(torch.tensor(reward))

        stacked_transposed_rewards = torch.stack(transposed_rewards, dim=0)
        assert stacked_transposed_rewards.shape == (
            len(transposed_reward_names),
            len(dataset),
        ), (
            f"unexpected rewards shape {stacked_transposed_rewards.shape},"
            f" expected ({len(transposed_reward_names)}, {len(dataset)})"
        )
        mean_rewards_by_type = stacked_transposed_rewards.mean(dim=1)
        for name, r in zip(transposed_reward_names, mean_rewards_by_type):
            logger.add_scalar.remote(
                f"test/{extra_reward_key}mean_reward_{name}", r.item(), step
            )
        table = [["index", "prompt", "reward", "completion"]]
        for index, completion in enumerate(completions):
            table.append(
                [
                    index,
                    json.dumps(dataset[index]),
                    stacked_transposed_rewards[:, index].tolist(),
                    completion,
                ]
            )
        logger.add_table.remote(f"eval{extra_table_key}", table, step)

        rewards = torch.stack(transposed_rewards, dim=1)
        agg_rewards = rewards.sum(dim=1)
        return agg_rewards


    def _get_reward_func(self, dataset_name: str) -> BunchedRewardFunction:
        if not self._many_reward_funcs:
            return self._reward_funcs
        else:
            return self._reward_funcs[dataset_name]
