from dataclasses import dataclass
from itertools import chain
import os
from pathlib import Path

from loguru import logger
from vllm import LLM, RequestOutput, CompletionOutput, SamplingParams


@dataclass(kw_only=True)
class ModelHandle():
    """
    A ModelHandle should be everything needed to generate with a model.

    This includes any default settings used for the generation.
    """
    llm: LLM
    default_sampling_params: SamplingParams
    prefix_system_prompts: list[str]


@dataclass
class PromptAndKey():
    prompt: str
    log_key: str


@dataclass
class ModelGenerateResult():
    output: str
    finish_reason: str | None
    stop_reason: int | str | None


def make_model_handle(
    model_ref: str | Path,
    default_sampling_params: SamplingParams,
    prefix_system_prompts: list[str] = [],
    extra_kwargs: dict = {},
) -> ModelHandle:
    model = LLM(
        model=str(model_ref),
        **extra_kwargs,
    )
    return ModelHandle(
        llm=model,
        default_sampling_params=default_sampling_params,
        prefix_system_prompts=prefix_system_prompts,
    )


def model_generate(
    prompts: list[PromptAndKey],
    api: ModelHandle,
    icl_shots: list[tuple[str, str]] = [],
    system_prompts: list[str] = [],
    override_sampling_params: SamplingParams | None = None,
) -> list[list[ModelGenerateResult]]:
    """
    Generates *chat* completions based on the given prompts.

    The given prompts are all batched together.

    ## Implementation notes
    This function could take extra arguments to control which defaults from ModelHandle are used,
    e.g., it could take a SamplingParams argument which would override the default sampling params.
    (To override only specific params the defaults should be cloned and mutated.)
    """
    try:
        output_samples = vllm_model_generate(
            prompts=prompts,
            api=api,
            icl_shots=icl_shots,
            system_prompts=system_prompts,
            override_sampling_params=override_sampling_params,
        )

        results: list[list[ModelGenerateResult]] = []
        for rd, single_prompt_samples in zip(prompts, output_samples):
            single_prompt_results = []
            for o in single_prompt_samples:
                if o.finish_reason == 'length':
                    logger.warning('Completion length limit reached at key: {!r}', rd.log_key)

                single_prompt_results.append(ModelGenerateResult(
                    output=o.text,
                    finish_reason=o.finish_reason,
                    stop_reason=o.stop_reason,
                ))
            results.append(single_prompt_results)

        return results
    except Exception:
        logger.exception('Exn at keys: {!r}', [rd.log_key for rd in prompts])
        return []


def vllm_model_generate(
    prompts: list[PromptAndKey],
    api: ModelHandle,
    icl_shots: list[tuple[str, str]] = [],
    system_prompts: list[str] = [],
    override_sampling_params: SamplingParams | None = None,
) -> list[list[CompletionOutput]]:
    base_messages = []
    for p in chain(api.prefix_system_prompts, system_prompts):
        base_messages.append({'role': 'system', 'content': p})
    for shot_prompt, shot_response in icl_shots:
        base_messages.append({'role': 'user', 'content': shot_prompt})
        base_messages.append({'role': 'assistant', 'content': shot_response})
    message_lists = []
    for rd in prompts:
        messages = base_messages.copy()
        messages.append({'role': 'user', 'content': rd.prompt})
        message_lists.append(messages)

    hack = os.getenv('AGNOSTICS_USE_VLLM_TQDM')
    use_tqdm = bool(hack)

    responses: list[RequestOutput] = api.llm.chat(
        messages=message_lists,
        use_tqdm=use_tqdm,
        sampling_params=override_sampling_params or api.default_sampling_params,
    )

    num_responses = len(responses)
    expected_num_responses = len(prompts)
    if num_responses != expected_num_responses:
        keys = [rd.log_key for rd in prompts]
        logger.error(
            "Error in code at keys: {!r}: expected {} RequestOutput, got {}. "
            "Responses will be mismatched with prompts.",
            keys,
            expected_num_responses,
            num_responses,
        )

    results: list[list[CompletionOutput]] = []
    for (rd, response) in zip(prompts, responses):
        results.append(response.outputs)

    return results
