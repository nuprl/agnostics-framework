"""
This module is only here to delay loading vllm until it's actually needed.
"""
from typing import Collection, Iterator

from vllm import SamplingParams

from . import vllm_facade

def gen_batch_output_rows(
    api: vllm_facade.ModelHandle,
    in_rows_batch: Collection[dict],
    icl_shots: list[tuple[str, str]] = [],
    system_prompts: list[str] = [],
    override_sampling_params: SamplingParams | None = None,
) -> Iterator[dict]:
    """
    Process a batch of input rows into a batch of output rows.

    Careful: `override_sampling_params` entirely override the defaults.
    To only override some of the defaults, clone and modify them.
    """
    requests = [vllm_facade.PromptAndKey(
        prompt=r['prompt'],
        log_key=str(r['idx']),
    ) for r in in_rows_batch]
    all_prompt_results = vllm_facade.model_generate(
        requests,
        api=api,
        icl_shots=icl_shots,
        system_prompts=system_prompts,
        override_sampling_params=override_sampling_params,
    )

    for in_r, one_prompt_results in zip(in_rows_batch, all_prompt_results):
        for sample_idx, res in enumerate(one_prompt_results):
            r = {
                'idx': in_r['idx'],
                'sample_idx': sample_idx,
                'prompt': in_r['prompt'],
                'response': res.output,
            }
            yield r