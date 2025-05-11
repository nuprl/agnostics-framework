print("Importing modules...")
import datasets
import ray
import time
from typing import List, Dict, Any, Generator, Tuple, Optional
from prl_ml.grpo.vanilla_grpo_trainer import Trainer
from prl_ml.grpo.logger import Logger, init_logger
from prl_ml.grpo.output_generator import OutputGenerator, GroupGenerationResult
print("Modules imported.")




class GRPOTester:
    def __init__(self):
        self.model_name = 'Qwen/Qwen2-0.5B-Instruct'
        
        max_len = max(len(name) for name in dir(self) if name.startswith("test_"))
        for name in dir(self):
            if name.startswith("test_"):
                print("="*80)
                print(f"Running   {name:<{max_len}}")
                print("-"*80)
                t_start = time.monotonic()
                getattr(self, name)()
                t_end = time.monotonic()
                print(f"Completed {name:<{max_len}} in {t_end - t_start:.3f} seconds.")
                print("="*80)
    
    def prompt_builder(self, item: Dict[str, Any], completion: Optional[str]):
        task = item["prompt"]
        prompt = task + self.prompt_suffix
        return prompt
    
    def reward_func10(self, items: Generator[Tuple[str, Dict[str, Any]], None, None]) -> List[float]:
        rewards = [10 * (int(item["id"]) + 42) for _, item in items]
        return rewards
    
    def reward_func20(self, items: Generator[Tuple[str, Dict[str, Any]], None, None]) -> List[float]:
        rewards = [20 * (int(item["id"]) + 42) for _, item in items]
        return rewards

    def test_01_dataset(self):
        self.prompt_suffix = "\ndef program(lst):\n"
        self.prompts = [
            "Write a python program to add all the odd numbers in a list.",
            "Write a python program to return a list of just the odd numbers in a list.",
            "Write a python program to return the standard deviation of a list of numbers.",
        ]
        dataset_list = [ {"prompt": prompt, "id": i} for i, prompt in enumerate(self.prompts) ]
        self.train_data = datasets.Dataset.from_list(dataset_list)
        self.test_data = datasets.Dataset.from_list(dataset_list)

    def test_02_creation(self):
        self.logger = init_logger("run", "Test", verbose=False)
        self.group_size = 4
        self.batch_size = 2
        self.trainer = Trainer.remote(
            model_name=self.model_name,
            train_data=self.train_data,
            reward_funcs=[self.reward_func10, self.reward_func20],
            prompt_builder=self.prompt_builder,
            stop=None,
            group_size=self.group_size,
            batch_size=self.batch_size,
            num_epochs=1,
            learning_rate=5e-6,
            group_sample_temperature=1.0,
            beta=0.0,
            group_max_tokens=128,
            vllm_gpu_memory_utilization=0.1,
            clipping_epsilon=0.1,
            logger=self.logger
        )
        ray.get(self.trainer.is_initialized.remote())
        
    def test_03_generation(self):
        print("Starting generation...")
        self.batch = self.train_data.select(range(self.batch_size))
        self.trainer.start_generation.remote(self.batch, 0)
        print("Waiting for generation to finish...")
        self.output_groups = self.trainer.finish_generation.remote()
        self.output_groups = ray.get(self.output_groups)
        assert len(self.output_groups) == self.batch_size
        for result in self.output_groups:
            assert len(result.outputs) == self.group_size
    
    def test_04_reward_computation(self):
        self.rewards = self.trainer.compute_rewards.remote(
            self.batch,
            self.output_groups,
            0
        )
        self.rewards = ray.get(self.rewards)
        assert len(self.rewards) == 2
        for r in self.rewards:
            assert r.shape == (self.group_size, self.batch_size)
        
        rewards10 = self.rewards[0]
        rewards20 = self.rewards[1]
        
        for g in range(self.group_size):
            for b in range(self.batch_size):
                assert rewards10[g, b] == 10 * (b + 42)
                assert rewards20[g, b] == 20 * (b + 42)

    def test_05_advantage_computation(self):
        advantages = self.trainer.compute_advantages.remote(self.rewards)
        advantages = ray.get(advantages)
        assert (advantages == 0.0).all()
        
if __name__ == "__main__":
    tester = GRPOTester()