from datetime import datetime
import io
import ray
import itertools
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from prl_ml.train.weight_decay import get_params_for_scheduler
import datasets
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from .util import batches, save_model_and_tokenizer
from .output_generator import OutputGenerator, GroupGenerationResult
from .evaluator import Evaluator
from pathlib import Path
from .logger import Logger, init_logger
from prl_ml.datasets.dataset_spec import DatasetSpec
from .types import PromptBuilder, RewardFunction, BunchedRewardFunction, ModelGenerationData
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from prl_ml.optimizers import get_optimizer
from prl_ml.schedulers import get_scheduler
import ray.util.collective as col
import time
from jaxtyping import Float, Int, Bool
# from accelerate import Accelerator # 8B


DEFAULT_TEST_FREQ = 10


@dataclass
class ReplayBufferItem:
    """
    Represents a single item in the replay buffer: a prompt and a list of outputs.
    """

    prompt_token_ids: Int[Tensor, "prompt_seq_len"]

    output_token_ids: List[Int[Tensor, "seq_len"]]

    advantages: Float[Tensor, "group_size"]

    def __init__(self,
        prompt_token_ids: Int[Tensor, "prompt_seq_len"],
        output_token_ids: List[Int[Tensor, "seq_len"]],
        advantages: Float[Tensor, "group_size"],
        item_len_limit: Optional[int] = None,
    ):
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = output_token_ids
        self.advantages = advantages
        self._clip_length(item_len_limit)


    def _clip_length(self, item_len_limit: Optional[int]) -> None:
        if item_len_limit is None:
            return
        max_output_len = item_len_limit - self.prompt_token_ids.shape[0]
        if max_output_len <= 0:
            self.prompt_token_ids = self.prompt_token_ids[:item_len_limit]
            self.output_token_ids = [ torch.empty(0, dtype=torch.int32) ] * len(self.output_token_ids)
            return

        self.output_token_ids = [output[:max_output_len] for output in self.output_token_ids]


def _advantage_product(
    scores: Float[Tensor, "batch_size seq_len"],
    advantages: Float[Tensor, "batch_size"],
) -> Float[Tensor, "batch_size seq_len"]:
    """Scale per-token *scores* by per-sequence *advantages*.

    Args:
        scores:    A (batch, seq) tensor.
        advantages: A (batch,)   tensor.
    Returns:
        The element-wise product with *advantages* broadcast across the
        sequence dimension. Shape is identical to *scores*.
    """
    return scores * advantages.reshape(scores.shape[0], 1)


def _compute_advantages(
    rewards: Float[Tensor, "group_size"],
) -> Tuple[Float[Tensor, "group_size"], float]:
    """
    Outcome supervision: advantage is the same for every token position.

    Returns the vector of advantages for each group item and the mean
    reward for across all groups.
    """
    rewards_std = rewards.std() + 1e-4
    advantages = (rewards - rewards.mean()) / rewards_std
    return advantages, rewards.mean().item()


def _log_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    replay_buffer_item: ReplayBufferItem,
    group_start: int,
    group_end: int,
) -> Tuple[Float[Tensor, "batch_size seq_len"], Bool[Tensor, "batch_size seq_len"]]:
    """
    Computes the logprobs of a subgroup of outputs.

    Notice that we do not use torch.no_grad() and model.eval(). This is
    deliberate: we want to use this in both train and eval mode.
    """
    subgroup_output_tokens = replay_buffer_item.output_token_ids[
        group_start:group_end
    ]
    padded_subgroup_output = pad_sequence(
        subgroup_output_tokens,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side="right",
    ).to(model.device)
    expanded_prompt_tokens = replay_buffer_item.prompt_token_ids.expand(
        padded_subgroup_output.shape[0], -1
    ).to(model.device)
    input_ids = torch.cat([expanded_prompt_tokens, padded_subgroup_output], dim=1)
    attention_mask = input_ids != tokenizer.pad_token_id
    logits = model(input_ids, attention_mask=attention_mask).logits
    prompt_len = replay_buffer_item.prompt_token_ids.shape[0]

    output_all_log_probs = F.log_softmax(logits[:, prompt_len - 1: -1, :], dim=-1)

    output_log_probs = torch.gather(
        output_all_log_probs, dim=-1, index=padded_subgroup_output.unsqueeze(-1)
    ).squeeze(-1)
    return output_log_probs, attention_mask[:, prompt_len:]


@ray.remote(num_gpus=1)
class Trainer:

    _logger: Logger

    def __init__(
        self,
        model_name: str,
        train_data: datasets.Dataset,
        train_item_len_limit: int,
        test_dataset_spec: str | dict[str, str],
        test_freq: int,
        replay_buffer_size: int,
        micro_batch_size: int,
        group_size: int,
        num_epochs: int,
        reward_func: BunchedRewardFunction,
        test_rewards_dict: dict[str, BunchedRewardFunction] | None,
        prompt_builder: PromptBuilder,
        stop: Optional[List[str]],
        vllm_gpu_memory_utilization: float,
        group_sample_temperature: float,
        clipping_epsilon: float,
        logger: Logger,
        concurrent_group_generation_hack: bool,
        beta: float,
        optimizer_str: str,
        scheduler_str: str,
        intermediate_checkpoint_freq: int,
        group_output_file: Optional[Path],
        output_generator: OutputGenerator,
        run_dir: str,
    ):
        """
        What is the concurrent group generation hack? In essence, instead of
        generating group outputs with the current model, we generate them with
        from the model before the latest update. This allows us to generate
        groups concurrently with model update. As long as the model does not
        update too much on each step, it should be close enough to using the
        current model.
        """
        ray.logger.info('Trainer initializing')
        self._logger = logger
        self._output_generator = output_generator
        output_generator_loaded = self._output_generator.load_model.remote()

        assert (
            group_size % micro_batch_size == 0
        ), f"group_size {group_size} must be divisible by micro_batch_size {micro_batch_size}"

        # self._accelerator = Accelerator() # 8B
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            # device_map="balanced" # 8B
        # ) # 8B
        ).to("cuda") # 4B
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.param_info = [(name, param.shape) for name, param in self._model.named_parameters()]
        self._train_data = train_data
        self._train_item_len_limit = train_item_len_limit
        self._test_freq = test_freq
        self._replay_buffer_size = replay_buffer_size
        self._micro_batch_size = micro_batch_size
        self._num_epochs = num_epochs
        self._group_size = group_size
        if self._tokenizer.pad_token_id is None:
            ray.logger.warning("WARNING: Using eos_token_id as pad_token_id.")
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._reward_func = reward_func
        self._prompt_builder = prompt_builder
        self._clipping_epsilon = clipping_epsilon
        self._steps_per_epoch = len(self._train_data) // self._replay_buffer_size
        self._concurrent_group_generation_hack = concurrent_group_generation_hack
        self._group_output_future = None
        self._evaluator = Evaluator.remote(
            output_generator=self._output_generator,
            test_dataset_spec=test_dataset_spec,
            reward_funcs=test_rewards_dict or reward_func,
            logger=self._logger,
        )
        self._optimizer_str = optimizer_str
        self._scheduler_str = scheduler_str
        self._beta = beta
        self._intermediate_checkpoint_freq = intermediate_checkpoint_freq

        if self._beta > 0.0:
            self._reference_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            ).to("cuda")
            for param in self._reference_model.parameters():
                param.requires_grad = False
            self._reference_model = self._reference_model.eval()
        else:
            self._reference_model = None
        ray.get(output_generator_loaded)
        ray.get(self._output_generator.set_param_info.remote(self.param_info))
        self._run_dir_path = Path(run_dir)

        ray.logger.info('Trainer initialized')
        _gb = 1024 ** 3
        for i in range(torch.cuda.device_count()):
            mem_total_gb = torch.cuda.mem_get_info(i)[1] / _gb
            self._logger.add_scalar.remote(f"resources/trainer/dev{i}/available_gb", mem_total_gb, 0)

    def rendezvous_send(self):
        waiting = self._output_generator.rendezvous_recv.remote()
        with torch.no_grad():
            for p in self._model.parameters():
                param_tensor = p.view(-1).cuda()
                col.send(param_tensor, 1)
        ray.get(waiting)

    def _log_probs(
        self,
        replay_buffer_item: ReplayBufferItem,
        ref_model: bool,
        group_start: int,
        group_end: int,
    ) -> Tuple[Float[Tensor, "batch_size seq_len"], Bool[Tensor, "batch_size seq_len"]]:
        return _log_probs(
            self._model if not ref_model else self._reference_model,
            self._tokenizer,
            replay_buffer_item,
            group_start,
            group_end,
        )

    def _rewards_by_group(
        self,
        batch_items: List[Dict[str, Any]],
        group_outputs: List[GroupGenerationResult],
        step: int,
    ) -> Float[Tensor, "batch_size group_size"]:
        """
        Receives a list of outputs from every batch item and returns a tensor
        of shape (batch_size, group_size). Each entry is the aggregate reward
        for a single output. Each row has the rewards for a group of outputs.
        """
        model_gen_data_list = [
            ModelGenerationData(output, item)
            for (i, item) in enumerate(batch_items)
            for output in group_outputs[i].outputs
        ]
        transposed_reward_names = []
        transposed_rewards = []
        for name, reward in self._reward_func.compute_reward(model_gen_data_list):
            transposed_reward_names.append(name)
            transposed_rewards.append(torch.tensor(reward))

        mean_rewards_by_type = torch.stack(transposed_rewards, dim=0).mean(dim=1)
        for name, r in zip(transposed_reward_names, mean_rewards_by_type):
            self._logger.add_scalar.remote(
                f"train/mean_reward_{name}", r.item(), step
            )
        flattened_rewards = torch.stack(transposed_rewards, dim=0).sum(dim=0)
        assert flattened_rewards.shape == (
            len(group_outputs) * self._group_size,
        ), f"unexpected flattened rewards shape {flattened_rewards.shape}"
        grouped_rewards = flattened_rewards.view((len(group_outputs), self._group_size))
        return grouped_rewards

    def _conclude_exploration(
        self,
        current_batch: List[Dict[str, Any]],
        current_batch_output_future: ray.ObjectRef,
        timer: ray.ObjectRef,
        step: int,
    ) -> List[ReplayBufferItem]:
        """
        Receives the group of outputs from vLLM for the current batch. Computes advantages
        and logprobs with the current model.
        """
        output_groups: List[GroupGenerationResult] = ray.get(
            current_batch_output_future
        )
        ray.get(timer.lap.remote("waiting_for_group_output"))
        agg_rewards = self._rewards_by_group(current_batch, output_groups, step)
        ray.get(timer.lap.remote("reward_computation"))
        next_replay_buffer: List[ReplayBufferItem] = []
        accumulated_rewards = []
        for group_index, item in enumerate(current_batch):
            this_group = output_groups[group_index]
            assert (
                len(this_group.outputs) == self._group_size
            ), f"Bug in OutputGenerator: got {len(this_group.outputs)} outputs for group size {self._group_size}"
            advantages, mean_rewards = _compute_advantages(
                agg_rewards[group_index].to(self._model.device)
            )
            replay_buffer_item = ReplayBufferItem(
                prompt_token_ids=this_group.prompt_token_ids,
                output_token_ids=this_group.output_token_ids,
                advantages=advantages,
                item_len_limit=self._train_item_len_limit,
            )
            next_replay_buffer.append(replay_buffer_item)
            accumulated_rewards.append(mean_rewards)
        self._logger.add_scalar.remote(
            "train/mean_reward", np.mean(accumulated_rewards), step
        )
        return next_replay_buffer

    def _concurrent_group_generation_hack_setup(
        self, step, current_batch, optimizer, lr_scheduler, max_steps: int,
    ):
        ts = datetime.now().strftime('%Y%m%dT%H%M%S')
        ray.logger.info(f"{ts} Start of step {step} / {max_steps}")
        if self._concurrent_group_generation_hack and self._group_output_future is None:
            self._group_output_future = (
                self._output_generator.sample_batched_group.remote(current_batch, step)
            )
            self._group_output_future_batch = current_batch
            return
        elif (
            self._concurrent_group_generation_hack
            and self._group_output_future is not None
        ):
            this_group_output_future = self._group_output_future
            this_group_batch = self._group_output_future_batch
            self._group_output_future = (
                self._output_generator.sample_batched_group.remote(current_batch, step)
            )
            self._group_output_future_batch = current_batch
        else:
            this_group_output_future = (
                self._output_generator.sample_batched_group.remote(current_batch, step)
            )
            this_group_batch = current_batch

        return self._train_inner_loop(
            step, this_group_batch, optimizer, lr_scheduler, this_group_output_future, max_steps
        )

    def _train_inner_loop(
        self, step, current_batch, optimizer, lr_scheduler, current_batch_output_future, max_steps,
    ):
        timer = ray.get(self._logger.create_timer.remote(step))

        self._logger.add_scalar.remote(
            "train/epoch", step / self._steps_per_epoch, step
        )

        self._model.eval()
        replay_buffer = self._conclude_exploration(
            current_batch, current_batch_output_future, timer, step
        )

        if step % self._test_freq == 0 or (step == 1 and self._concurrent_group_generation_hack):
            eval_future = self._evaluator.evaluate.remote(step)
        else:
            eval_future = None

        if self.should_save_intermediate_checkpoint(step, max_steps):
            ray.logger.info(f"Saving model and tokenizer at step {step}")
            save_model_and_tokenizer(self._model, self._tokenizer, self._run_dir_path / f"checkpoint_{step}")

        _gb = 1024 ** 3
        for i in range(torch.cuda.device_count()):
            mem_allocated_peak_gb = torch.cuda.max_memory_allocated(i) / _gb
            mem_reserved_peak_gb = torch.cuda.max_memory_reserved(i) / _gb
            self._logger.add_scalar.remote(f"resources/trainer/dev{i}/pre_loss_peak_allocated_gb", mem_allocated_peak_gb, step)
            self._logger.add_scalar.remote(f"resources/trainer/dev{i}/pre_loss_peak_reserved_gb", mem_reserved_peak_gb, step)
        torch.cuda.reset_peak_memory_stats()

        self._model.eval()
        accumulated_losses = []
        accumulated_kl = [ ]
        accumulated_advantages = [ ]

        accumulation_steps = (
            len(replay_buffer) * self._group_size
        ) // self._micro_batch_size


        self._logger.add_scalar.remote(
            "train/max_item_len",
            max(item.prompt_token_ids.shape[0] + output.shape[0] for item in replay_buffer for output in item.output_token_ids),
            step)


        for replay_buffer_item, group_start_index in itertools.product(
            replay_buffer, range(0, self._group_size, self._micro_batch_size)
        ):
            group_end_index = group_start_index + self._micro_batch_size

            advantages = replay_buffer_item.advantages[group_start_index:group_end_index]

            if (advantages.abs() < 1e-07).all().item():
                accumulated_losses.append(0.0)
                accumulated_kl.append(0.0)
                accumulated_advantages.append(0.0)
                continue

            replay_buffer_new_log_probs, completion_mask = self._log_probs(
                replay_buffer_item,
                ref_model=False,
                group_start=group_start_index,
                group_end=group_end_index,
            )

            if self._reference_model is not None:
                replay_buffer_ref_log_probs, _ = self._log_probs(
                    replay_buffer_item,
                    ref_model=True,
                    group_start=group_start_index,
                    group_end=group_end_index,
                )
                per_token_kl = (
                    torch.exp(replay_buffer_ref_log_probs - replay_buffer_new_log_probs)
                    - (replay_buffer_ref_log_probs - replay_buffer_new_log_probs)
                    - 1
                )
            else:
                per_token_kl = torch.zeros_like(
                    replay_buffer_new_log_probs,
                    device=replay_buffer_new_log_probs.device,
                )

            prob_ratio = torch.exp(
                replay_buffer_new_log_probs - replay_buffer_new_log_probs.detach()
            )

            raw_loss_2d = _advantage_product(prob_ratio, advantages)
            clamped_loss_2d = _advantage_product(
                torch.clamp(
                    prob_ratio,
                    min=1 - self._clipping_epsilon,
                    max=1 + self._clipping_epsilon,
                ),
                advantages,
            )

            loss_2d = (
                torch.min(raw_loss_2d, clamped_loss_2d) - self._beta * per_token_kl
            ) * completion_mask
            loss_1d = loss_2d.sum(dim=1) / completion_mask.sum(dim=1)
            loss = -loss_1d.mean() / accumulation_steps
            loss.backward()
            accumulated_losses.append(loss.item())
            accumulated_kl.append((per_token_kl * completion_mask).mean().item())
            accumulated_advantages.append(advantages.mean().item())

        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.1)
        ray.get(timer.lap.remote("main_loss_computation"))
        self._logger.add_scalar.remote("train/grad_norm", grad_norm.item(), step)
        self._logger.add_scalar.remote(
            "train/learning_rate", optimizer.param_groups[0]["lr"], step
        )
        self._logger.add_scalar.remote("train/kl", np.mean(accumulated_kl), step)
        self._logger.add_scalar.remote("train/advantages", np.mean(accumulated_advantages), step)

        if eval_future is not None:
            ray.get(eval_future)
            eval_future = None

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        self._logger.add_scalar.remote("train/loss", np.mean(accumulated_losses), step)
        ray.get(timer.lap.remote("main_model_update"))

        self.rendezvous_send()
        ray.get(timer.lap.remote("waiting_for_rdv"))
        ray.get(timer.done.remote())

    def train(self):
        steps_per_epoch = len(self._train_data) // self._replay_buffer_size
        max_steps = self._num_epochs * steps_per_epoch
        ray.logger.info(f"Max steps: {max_steps}")
        optimizer = get_optimizer(self._optimizer_str)(get_params_for_scheduler(self._model))

        # self._model, optimizer = self._accelerator.prepare(self._model, optimizer) # 8B
        lr_scheduler = get_scheduler(self._scheduler_str)(optimizer, max_steps)

        ray.logger.info(f"Starting training")
        batch_iter = batches(
            self._num_epochs, self._replay_buffer_size, self._train_data
        )
        for step, current_batch in enumerate(batch_iter):
            if step > 1 and (step-1) % steps_per_epoch == 0:
                epoch_num = (step-1) // steps_per_epoch
                ray.logger.info(f"Saving model and tokenizer at end of epoch {epoch_num}")
                save_model_and_tokenizer(
                    self._model,
                    self._tokenizer,
                    self._run_dir_path / f"checkpoint_epoch{epoch_num}",
                )

            self._concurrent_group_generation_hack_setup(
                step, current_batch, optimizer, lr_scheduler, max_steps=max_steps
            )

        if step % self._test_freq != 0:
            ray.get(self._evaluator.evaluate.remote(step))

        ray.logger.info("Saving model and tokenizer at final step")
        save_model_and_tokenizer(self._model, self._tokenizer, self._run_dir_path / "checkpoint_final")


    def should_save_intermediate_checkpoint(self, step: int, max_steps: int) -> bool:
        freq = self._intermediate_checkpoint_freq
        if freq == 0:
            return False
        return step > 1 and step != max_steps - 1 and step % freq == 0



def train(
    model_name: str,
    train_dataset_spec: str,
    test_dataset_spec: str | dict[str, str],
    run_dir: str,
    batch_size: int,
    micro_batch_size: int,
    num_epochs: int,
    group_size: int,
    reward_funcs: BunchedRewardFunction | List[RewardFunction],
    prompt_builder: PromptBuilder,
    stop: Optional[List[str]],
    vllm_gpu_memory_utilization: float,
    group_sample_temperature: float,
    group_max_tokens: int,
    clipping_epsilon: float,
    concurrent_group_generation_hack: bool,
    beta: float,
    group_output_file: Optional[Path],
    optimizer_str: str,
    scheduler_str: str,
    train_item_len_limit: int,
    train_data_shuffle: int | bool = True,
    custom_test_freq: int | None = None,
    test_rewards_dict: dict[str, BunchedRewardFunction] | None = None,
    checkpoint_freq: int = 0,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
):
    """
    Train the model with GRPO.

    Args:
        test_dataset_spec: str | dict[str, str]
            A str is resolved with DatasetSpec.from_string.
            A dict means there are multiple test datasets.
            The keys are dataset names, and the values are resolved the same way as a str would be.
        test_rewards_dict: dict[str, BunchedRewardFunction] | None
            Only meaningful if test_dataset_spec is a dict.
            A dict of reward functions, one per test dataset.
    """
    train_data = DatasetSpec.from_string(train_dataset_spec).load()
    if train_data_shuffle is not False:
        seed: int = train_data_shuffle if train_data_shuffle is not True else 42
        train_data = train_data.shuffle(seed=seed)

    if isinstance(test_dataset_spec, dict) and test_rewards_dict:
        missing_keys = set(test_dataset_spec.keys()) - set(test_rewards_dict.keys())
        if missing_keys:
            raise ValueError(f"Missing reward functions for test datasets: {', '.join(missing_keys)}")
        for spec in test_dataset_spec.values():
            DatasetSpec.from_string(spec).load()
    elif isinstance(test_dataset_spec, str):
        DatasetSpec.from_string(test_dataset_spec).load()

    test_freq = custom_test_freq or DEFAULT_TEST_FREQ
    assert test_freq > 0

    assert train_item_len_limit > 0, "train_item_len_limit must be positive"

    assert checkpoint_freq >= 0

    hyperparameters = {
        "model_name": model_name,
        "train_dataset_spec": train_dataset_spec,
        "test_dataset_spec": test_dataset_spec,
        "train_item_len_limit": train_item_len_limit,
        "test_freq": test_freq,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "num_epochs": num_epochs,
        "group_size": group_size,
        "stop": stop if stop is not None else [ ],
        "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
        "group_sample_temperature": group_sample_temperature,
        "clipping_epsilon": clipping_epsilon,
        "concurrent_group_generation_hack": concurrent_group_generation_hack,
        "beta": beta,
        "optimizer": optimizer_str,
        "scheduler": scheduler_str,
        "checkpoint_freq": checkpoint_freq,
    }

    logger = init_logger(
        run_dir=run_dir,
        hyperparameters=hyperparameters,
        project_name=project_name,
        run_name=run_name,
    )

    ray.init(namespace="transfer_ns", ignore_reinit_error=True)
    output_generator = OutputGenerator.remote(
            init_model_path=model_name,
            group_size=group_size,
            group_max_tokens=group_max_tokens,
            group_sample_temperature=group_sample_temperature,
            prompt_builder=prompt_builder,
            stop=stop,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            logger=logger,
            group_output_file=group_output_file,
        )

    if isinstance(reward_funcs, list):
        reward_funcs = BunchedRewardFunction.from_func_list(reward_funcs)

    trainer = Trainer.remote(
        model_name=model_name,
        train_data=train_data,
        train_item_len_limit=train_item_len_limit,
        micro_batch_size=micro_batch_size,
        replay_buffer_size=batch_size,
        group_size=group_size,
        num_epochs=num_epochs,
        test_dataset_spec=test_dataset_spec,
        test_freq=test_freq,
        reward_func=reward_funcs,
        test_rewards_dict=test_rewards_dict,
        prompt_builder=prompt_builder,
        stop=stop,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        group_sample_temperature=group_sample_temperature,
        clipping_epsilon=clipping_epsilon,
        concurrent_group_generation_hack=concurrent_group_generation_hack,
        logger=logger,
        beta=beta,
        optimizer_str=optimizer_str,
        scheduler_str=scheduler_str,
        group_output_file=group_output_file,
        intermediate_checkpoint_freq=checkpoint_freq,
        output_generator=output_generator,
        run_dir=run_dir,
    )

    col.create_collective_group(
            [trainer,output_generator],
            world_size=2,
            ranks=[0, 1],
            backend="nccl",
        )
    ray.get(trainer.train.remote())
    ray.get(logger.close.remote())
    ray.shutdown()
