import ray
import copy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
import datasets
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

from prl_ml.datasets.dataset_spec import DatasetSpec
from prl_ml.train.weight_decay import get_params_for_scheduler

from .util import batches, save_model_and_tokenizer
from .output_generator import OutputGenerator, GroupGenerationResult
from .evaluator import Evaluator
from .logger import Logger, init_logger
from .types import PromptBuilder, RewardFunction


@ray.remote(num_gpus=1)
class Trainer:
    _logger: Logger

    def __init__(
        self,
        model_name: str,
        
        train_data: datasets.Dataset,
        reward_funcs: List[RewardFunction],
        prompt_builder: PromptBuilder,
        stop: Optional[List[str]],
        
        group_size: int,
        batch_size: int,
        num_epochs: int,
        
        learning_rate: float,
        group_sample_temperature: float,
        beta: float,
        group_max_tokens: int,
        
        vllm_gpu_memory_utilization: float,
        
        clipping_epsilon: float,
        logger: Logger,
    ):
        self.train_device = "cuda"
        self._logger = logger
        self._output_generator = OutputGenerator.remote(
            init_model_path=model_name,
            group_size=group_size,
            group_max_tokens=group_max_tokens,
            group_sample_temperature=group_sample_temperature,
            prompt_builder=prompt_builder,
            stop=stop,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            logger=self._logger
        )
        self._batch_size = batch_size
        self._group_max_tokens = group_max_tokens
        
        output_generator_loaded = self._output_generator.load_model.remote()

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        
        self._prev_model = copy.deepcopy(self._model)
        
        self._train_data = train_data
        self._learning_rate = learning_rate
        self._num_epochs = num_epochs
        self._group_size = group_size
        if self._tokenizer.pad_token_id is None:
            print("WARNING: Using eos_token_id as pad_token_id.")
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._reward_funcs = reward_funcs
        self._prompt_builder = prompt_builder
        self._clipping_epsilon = clipping_epsilon
        self._group_output_future = None

        self._beta = beta
        if self._beta > 0.0:
            raise NotImplementedError("Beta > 0.0 is not supported yet.")
          
        print("Waiting for output generator to load...")
        ray.get(output_generator_loaded)
        print("Output generator loaded.")
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def compute_rewards(self,
        batch_items: List[Dict[str, Any]],
        group_outputs: List[GroupGenerationResult],
        step: int
    ) -> List[torch.Tensor]:
        assert len(batch_items) == len(group_outputs)
        outputs_with_items = []
        for i, item in enumerate(batch_items):
            for output in group_outputs[i].outputs:
                outputs_with_items.append((output, item))

        rewards = []
        for reward_func in self._reward_funcs:
            reward = torch.tensor(reward_func(outputs_with_items)).to(self.train_device)
            reward = reward.reshape(self._batch_size, self._group_size).T
            rewards.append(reward)
        
        return rewards

    def compute_advantages(self, rewards: List[torch.Tensor]) -> torch.Tensor:
        rewards = torch.sum(torch.stack(rewards), dim=0).to(torch.float32)
        
        rewards_std, rewards_mean = torch.std_mean(rewards, dim=0)
        
        rewards_mean = rewards_mean.unsqueeze(0).repeat(self._group_size, 1)
        rewards_std = rewards_std.unsqueeze(0).repeat(self._group_size, 1)
        
        advantages = (rewards - rewards_mean) / rewards_std
        advantages[rewards_std < 1e-10] = 0.0
        return advantages
      
    def compute_loss(self, 
                      output_groups: List[GroupGenerationResult], 
                      advantages: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1)
        return loss
    
    def start_generation(self, current_batch, step_i):
        assert len(current_batch) <= self._batch_size
        self._group_output_future = self._output_generator.sample_batched_group.remote(
                current_batch, step_i)
    
    def finish_generation(self):
        output_groups: List[GroupGenerationResult] = ray.get(self._group_output_future)
        return output_groups
      
    def train(self):
        optimizer = Adam(
            get_params_for_scheduler(self._model),
            weight_decay=0.1,
            lr=self._learning_rate,
            betas=(0.9, 0.99),
        )
        
        max_steps = len(self._train_data) * self._num_epochs // self._batch_size
        
        print(f"Max steps: {max_steps}")
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=0.1 * max_steps,
            num_training_steps=max_steps,
        )

        batch_iter = batches(
            self._num_epochs, self._batch_size, self._train_data
        )
        for step_i, current_batch in enumerate(batch_iter):
            print(f"Start of step {step_i}")
          
            self.start_generation(current_batch, step_i)
            
            output_groups: List[GroupGenerationResult] = self.finish_generation()
            
            
            rewards: List[torch.Tensor] = self.compute_rewards(output_groups)
            
            advantages: torch.Tensor = self.compute_advantages(rewards)
            
            loss = self.compute_loss(output_groups, advantages)
                        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            
            

        if step_i % 10 != 0:
            ray.get(self._evaluator.evaluate.remote(step_i))

        print("Saving model and tokenizer")
        save_model_and_tokenizer(self._model, self._tokenizer, Path("checkpoint_final"))


def train(
    model_name: str,
    train_dataset_spec: str,
    test_dataset_spec: str,
    run_dir: str,
    learning_rate: float,
    batch_size: int,
    micro_batch_size: int,
    num_epochs: int,
    group_size: int,
    reward_funcs: List[RewardFunction],
    prompt_builder: PromptBuilder,
    stop: Optional[List[str]],
    vllm_gpu_memory_utilization: float,
    group_sample_temperature: float,
    clipping_epsilon: float,
    concurrent_group_generation_hack: bool,
    beta: float,
    group_output_file: Optional[Path],
    project_name: Optional[str] = None,
):
    train_data = DatasetSpec.from_string(train_dataset_spec).load().shuffle(seed=42)
    logger = init_logger(run_dir, project_name)
    trainer = Trainer.remote(
        model_name=model_name,
        train_data=train_data,
        micro_batch_size=micro_batch_size,
        replay_buffer_size=batch_size,
        group_size=group_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        test_dataset_spec=test_dataset_spec,
        reward_funcs=reward_funcs,
        prompt_builder=prompt_builder,
        stop=stop,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        group_sample_temperature=group_sample_temperature,
        clipping_epsilon=clipping_epsilon,
        concurrent_group_generation_hack=concurrent_group_generation_hack,
        logger=logger,
        beta=beta,
        group_output_file=group_output_file,
    )
    ray.get(trainer.train.remote())
    ray.shutdown()
    
    