import io
import ray
from vllm import LLM, SamplingParams
from typing import List, Dict, Tuple, Any, Optional
import time
import torch
from .types import PromptBuilder
from pathlib import Path
import json
import ray.util.collective as col
from dataclasses import dataclass
NUM_GPUS = 1

@dataclass
class GroupGenerationResult:
    prompt_token_ids: torch.Tensor
    outputs: List[str]
    output_token_ids: List[torch.Tensor]


@ray.remote(num_gpus=NUM_GPUS)
class OutputGenerator:
    """
    A Ray Actor that uses vLLM under the hood that is designed to be used in a
    GRPO training loop. It supports sampling a group of completions, generation
    for the purpose of evaluation, and updating the model weights.
    """

    def __init__(
        self,
        init_model_path: str,
        group_size: int,
        group_max_tokens: int,
        group_sample_temperature: float,
        prompt_builder: PromptBuilder,
        gpu_memory_utilization: float,
        stop: Optional[List[str]],
        logger: ray.ObjectRef,
        group_output_file: Optional[Path] = None,
    ):
        self._init_model_path = init_model_path
        self._fully_initialized = False
        self._model = None
        self._prompt_builder = prompt_builder
        self._logger = logger
        self.gpu_memory_utilization = gpu_memory_utilization

        self._sampling_params = SamplingParams(
            temperature=group_sample_temperature,
            top_p=0.95,
            max_tokens=group_max_tokens,
            n=group_size,
            stop=stop,
        )

        self._group_output_file = group_output_file.open("a") if group_output_file is not None else None

    def set_param_info(self, info):
        self.param_info = info

    def rendezvous_recv(self):
        state_dict = {}
        for name, param_shape in self.param_info:
            param_size = torch.prod(torch.tensor(param_shape)).item()
            param_buffer = torch.empty(param_size, dtype=torch.bfloat16, device="cuda")
            col.recv(param_buffer, 0)
            state_dict[name] = param_buffer.view(param_shape)
        self.update_model(state_dict)

    def load_model(self):
        assert not self._fully_initialized, "Model must be loaded only once"
        self._model = LLM(
            model=self._init_model_path,
            dtype=torch.bfloat16,
            tensor_parallel_size=NUM_GPUS,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self._fully_initialized = True
        ray.logger.info('Model loaded')

        _gb = 1024 ** 3
        for i in range(torch.cuda.device_count()):
            mem_total_gb = torch.cuda.mem_get_info(i)[1] / _gb
            self._logger.add_scalar.remote(f"resources/generator/dev{i}/available_gb", mem_total_gb, 0)

    def _chat_or_base(
        self,
        prompts: List[str] | List[Dict[str, Any]],
        sp: SamplingParams,
    ) -> list:
        """
        VLLM has two different methods for chat and base generation. This
        function just calls the appropriate method by testing if the first
        prompt is a string (base generation) or a dict (chat).
        """
        if type(prompts[0]) != str:
            return self._model.chat(prompts, sp, use_tqdm=False)
        else:
            return self._model.generate(prompts, sp, use_tqdm=False)

    def generate(self, prompts: List[Dict[str, Any]], temperature: float, top_p: float = 1.0) -> List[str]:
        assert self._fully_initialized, "OutputGenerator is not initialized"
        prompts = [self._prompt_builder(p, None) for p in prompts]
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=self._sampling_params.max_tokens,
            stop=self._sampling_params.stop,
        )
        outputs = []
        completions = self._chat_or_base(prompts, sp)
        for completion in completions:
            assert len(completion.outputs) == 1
            outputs.append(completion.outputs[0].text)
        return outputs

    def _may_log(self, json_dict):
        if self._group_output_file is not None:
            self._group_output_file.write(json.dumps(json_dict) + "\n")
            self._group_output_file.flush()


    def sample_batched_group(self, items: List[Dict[str, Any]], step: int) -> List[GroupGenerationResult]:
        assert self._fully_initialized, "OutputGenerator is not initialized"
        t_start = time.perf_counter()

        _gb = 1024 ** 3
        torch.cuda.reset_peak_memory_stats()
        for i in range(torch.cuda.device_count()):
            mem_allocated_gb = torch.cuda.memory_allocated(i) / _gb
            mem_reserved_gb = torch.cuda.memory_reserved(i) / _gb
            self._logger.add_scalar.remote(f"resources/generator/dev{i}/pre_batch_allocated_gb", mem_allocated_gb, step)
            self._logger.add_scalar.remote(f"resources/generator/dev{i}/pre_batch_reserved_gb", mem_reserved_gb, step)

        prompts = [self._prompt_builder(item, None) for item in items]
        batch_completions = self._chat_or_base(prompts, self._sampling_params)
        results = []
        for completion, prompt in zip(batch_completions, prompts):
            this_result = GroupGenerationResult(
                prompt_token_ids=torch.tensor(completion.prompt_token_ids, dtype=torch.long),
                outputs=[item.text for item in completion.outputs],
                output_token_ids=[torch.tensor(item.token_ids, dtype=torch.long) for item in completion.outputs]
            )
            results.append(this_result)
            self._may_log({"prompt": prompt, "outputs": this_result.outputs, "step": step})

        for i in range(torch.cuda.device_count()):
            mem_allocated_peak_gb = torch.cuda.max_memory_allocated(i) / _gb
            mem_reserved_peak_gb = torch.cuda.max_memory_reserved(i) / _gb
            self._logger.add_scalar.remote(f"resources/generator/dev{i}/post_batch_peak_allocated_gb", mem_allocated_peak_gb, step)
            self._logger.add_scalar.remote(f"resources/generator/dev{i}/post_batch_peak_reserved_gb", mem_reserved_peak_gb, step)

        self._logger.add_scalar.remote("timer/output_generation", time.perf_counter() - t_start, step)
        return results

    def update_model(self, state_dict: Dict[str, torch.Tensor]):
        """
        Updates the model weights in-place. This should receive the state_dict
        from the trainer after optimizer.step().
        """
        model = self._model.llm_engine.model_executor.driver_worker.model_runner.model
        model.load_weights(state_dict.items())
