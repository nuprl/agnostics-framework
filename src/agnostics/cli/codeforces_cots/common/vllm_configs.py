from dataclasses import dataclass, field
from pathlib import Path

from vllm import SamplingParams

from .vllm_facade import ModelHandle, make_model_handle
from ... import cmd


REASONABLE_MAX_TOKENS = 12*1024


@dataclass
class ModelConfig:
    config_key: str
    default_sampling_params: SamplingParams
    prefix_system_prompts: list[str] = field(default_factory=list)
    extra_kwargs: dict = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    model_ref: str | Path
    model_config: ModelConfig
    extra_kwargs: dict = field(default_factory=dict)

    def as_model_handle_kwargs(self) -> dict:
        """
        Returns a dict of kwargs which can be passed to `make_model_handle`.
        """
        extra_kwargs = self.model_config.extra_kwargs.copy()
        extra_kwargs.update(self.extra_kwargs)
        res = {
            'model_ref': self.model_ref,
            'default_sampling_params': self.model_config.default_sampling_params,
            'prefix_system_prompts': self.model_config.prefix_system_prompts,
            'extra_kwargs': extra_kwargs,
        }
        return res


def _qwen3_nothinks_model_config(
    config_key: str,
    max_tokens: int = REASONABLE_MAX_TOKENS,
) -> ModelConfig:
    return ModelConfig(
        config_key=config_key,
        default_sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
            max_tokens=max_tokens,
        ),
        prefix_system_prompts=['/nothink'],
    )


def _qwen3_thinks_model_config(config_key: str) -> ModelConfig:
    cfg = _qwen3_nothinks_model_config(config_key)
    cfg.prefix_system_prompts.clear()
    return cfg


def _smollm3_nothinks_model_config(
    config_key: str,
    max_tokens: int = REASONABLE_MAX_TOKENS,
) -> ModelConfig:
    return ModelConfig(
        config_key=config_key,
        default_sampling_params=SamplingParams(
            temperature=0.6,
            top_p=0.8,
            top_k=20,
            min_p=0,
            max_tokens=max_tokens,
        ),
        prefix_system_prompts=['/no_think'],
    )


def _smollm3_thinks_model_config(
    config_key: str,
    max_tokens: int = REASONABLE_MAX_TOKENS,
) -> ModelConfig:
    return ModelConfig(
        config_key=config_key,
        default_sampling_params=SamplingParams(
            temperature=0.6,
            top_p=0.8,
            top_k=20,
            min_p=0,
            max_tokens=max_tokens,
        ),
    )


def _hunyuan7b_nothinks_model_config(
    config_key: str,
    max_tokens: int = REASONABLE_MAX_TOKENS,
) -> ModelConfig:
    return ModelConfig(
        config_key=config_key,
        default_sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_tokens=max_tokens,
        ),
        prefix_system_prompts=['/no_think'],
    )


ENV_CONFIGS = {
    'lcbx--smollm3': EnvironmentConfig(
        model_ref='HuggingFaceTB/SmolLM3-3B',
        model_config=ModelConfig(
            config_key='lcbx--smollm3',
            default_sampling_params=SamplingParams(
                max_tokens=5*1024,
            ),
            prefix_system_prompts=['/no_think'],
        ),
    ),
    'lcbx--dscoder-6p7b': EnvironmentConfig(
        model_ref='deepseek-ai/deepseek-coder-6.7b-instruct',
        model_config=ModelConfig(
            config_key='lcbx--dscoder-6p7b',
            default_sampling_params=SamplingParams(
                max_tokens=5*1024,
            ),
        ),
        extra_kwargs=cmd.typecheck_jsonable({
            'max_model_len': 30000,
        }),
    ),
    'lcbx--rnj1': EnvironmentConfig(
        model_ref='EssentialAI/rnj-1-instruct',
        model_config=ModelConfig(
            config_key='lcbx--rnj1',
            default_sampling_params=SamplingParams(max_tokens=5*1024),
        ),
    ),
    'lcbx--hunyuan7b': EnvironmentConfig(
        model_ref='tencent/Hunyuan-7B-Instruct',
        model_config=_hunyuan7b_nothinks_model_config('lcbx--hunyuan7b', max_tokens=3*1024),
    ),

    'qwen3-4B': EnvironmentConfig(
        model_ref='Qwen/Qwen3-4B',
        model_config=_qwen3_nothinks_model_config('qwen3-4B'),
    ),
    'qwen3-4B+thinks': EnvironmentConfig(
        model_ref='Qwen/Qwen3-4B',
        model_config=_qwen3_thinks_model_config('qwen3-4B__thinks'),
    ),
    'qwen3-4B+max-tokens-5k': EnvironmentConfig(
        model_ref='Qwen/Qwen3-4B',
        model_config=_qwen3_nothinks_model_config('qwen3-4B__max-tokens-5k', max_tokens=5*1024),
    ),
    'qwen3-1.7B': EnvironmentConfig(
        model_ref='Qwen/Qwen3-1.7B',
        model_config=_qwen3_nothinks_model_config('qwen3-1p7B'),
    ),
    'qwen3-1.7B+max-tokens-5k': EnvironmentConfig(
        model_ref='Qwen/Qwen3-1.7B',
        model_config=_qwen3_nothinks_model_config('qwen3-1p7B__max-tokens-5k', max_tokens=5*1024),
    ),
    'qwen3-1.7B+thinks': EnvironmentConfig(
        model_ref='Qwen/Qwen3-1.7B',
        model_config=_qwen3_thinks_model_config('qwen3-1p7B__thinks'),
    ),

    'smollm3-3B+max-tokens-5k': EnvironmentConfig(
        model_ref='HuggingFaceTB/SmolLM3-3B',
        model_config=_smollm3_nothinks_model_config('smollm3-3B__max-tokens-5k', max_tokens=5*1024),
    ),
    'smollm3-3B+thinks+max-tokens-5k': EnvironmentConfig(
        model_ref='HuggingFaceTB/SmolLM3-3B',
        model_config=_smollm3_thinks_model_config('smollm3-3B__thinks_max-tokens-5k', max_tokens=5*1024),
    ),

    'alex/boa/qwen3-32B': EnvironmentConfig(
        model_ref=Path.home()/'models/Qwen3-32B',
        model_config=_qwen3_nothinks_model_config('qwen3-32B'),
        extra_kwargs=cmd.typecheck_jsonable({ 'max_model_len': 10240, }),
    ),
    'alex/boa/qwen3-14B': EnvironmentConfig(
        model_ref=Path.home()/'models/Qwen3-14B',
        model_config=_qwen3_nothinks_model_config('qwen3-14B'),
    ),
    'alex/boa/qwen3-8B': EnvironmentConfig(
        model_ref=Path.home()/'models/Qwen3-8B',
        model_config=_qwen3_nothinks_model_config('qwen3-8B'),
    ),
    'alex/boa/qwen3-4B': EnvironmentConfig(
        model_ref=Path.home()/'models/Qwen3-4B',
        model_config=_qwen3_nothinks_model_config('qwen3-4B'),
    ),
    'alex/boa/qwen3-4B+thinks': EnvironmentConfig(
        model_ref=Path.home()/'models/Qwen3-4B',
        model_config=_qwen3_thinks_model_config('qwen3-4B__thinks'),
    ),
    'alex/boa/qwen3-1.7B': EnvironmentConfig(
        model_ref=Path.home()/'models/Qwen3-1p7B',
        model_config=_qwen3_nothinks_model_config('qwen3-1p7B'),
    ),
    'alex/boa/qwen3-1.7B+thinks': EnvironmentConfig(
        model_ref=Path.home()/'models/Qwen3-1p7B',
        model_config=_qwen3_thinks_model_config('qwen3-1p7B__thinks'),
    ),
    'alex/boa/qwen3-1.7B+max-tokens-5k': EnvironmentConfig(
        model_ref=Path.home()/'models/Qwen3-1p7B',
        model_config=_qwen3_nothinks_model_config('qwen3-1p7B__max-tokens-5k', max_tokens=5*1024),
    ),
    'alex/boa/qwen2p5-coder-14B': EnvironmentConfig(
        model_ref=Path.home()/'models/Qwen2p5-Coder-14B-Instruct',
        model_config=ModelConfig(
            config_key='qwen2p5-coder-14B',
            default_sampling_params=SamplingParams(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0,
                max_tokens=REASONABLE_MAX_TOKENS,
            ),
        )
    ),
}


def model_from_env_cfg_name(env: str) -> ModelHandle:
    cfg = ENV_CONFIGS[env]
    return model_from_config(cfg)


def model_from_config(cfg: EnvironmentConfig) -> ModelHandle:
    extra_kwargs = cfg.model_config.extra_kwargs.copy()
    extra_kwargs.update(cfg.extra_kwargs)
    return make_model_handle(
        model_ref=cfg.model_ref,
        default_sampling_params=cfg.model_config.default_sampling_params,
        prefix_system_prompts=cfg.model_config.prefix_system_prompts,
        extra_kwargs=extra_kwargs,
    )