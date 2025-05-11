"""
Some helpers functions for StarCoder models.
"""
from transformers import AutoModelForCausalLM


def _get_architecture(model: AutoModelForCausalLM) -> str:
    return model.config.architectures[0]


def _smollm3_params_for_scheduler(model, weight_decay):
    """
    We use this to configure weight decay for SmolLM3 models.
    (No weight decay for biases or *_layernorm.weight parameters.)
    """
    params_with_wd = []
    params_without_wd = []
    for n, p in model.named_parameters():
        if "bias" in n:
            params_without_wd.append(p)
        elif "_layernorm" in n and "weight" in n:
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def _phi3_params_for_scheduler(model, weight_decay):
    params_with_wd = []
    params_without_wd = []
    for n, p in model.named_parameters():
        if "bias" in n:
            params_without_wd.append(p)
        elif "norm" in n and "weight" in n:
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def _olmo2_params_for_scheduler(model, weight_decay):
    """
    We use this to configure weight decay for Olmo2 models.
    """
    params_with_wd = []
    params_without_wd = []
    for n, p in model.named_parameters():
        if "bias" in n:
            params_without_wd.append(p)
        elif "norm" in n and "weight" in n:
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def _qwen2_params_for_scheduler(model, weight_decay):
    """
    We use this to configure weight decay for Qwen2 models.
    """
    params_with_wd = []
    params_without_wd = []
    for n, p in model.named_parameters():
        if "bias" in n:
            params_without_wd.append(p)
        elif "norm" in n and "weight" in n:
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def _qwen3_params_for_scheduler(model, weight_decay):
    """
    We use this to configure weight decay for Qwen2 models.
    """
    params_with_wd = []
    params_without_wd = []
    for n, p in model.named_parameters():
        if "bias" in n:
            params_without_wd.append(p)
        elif "norm" in n and "weight" in n:
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def _llama3_params_for_scheduler(model, weight_decay):
    """
    We use this to configure weight decay for Llama3 models.
    """
    params_with_wd = []
    params_without_wd = []
    for n, p in model.named_parameters():
        if "bias" in n:
            params_without_wd.append(p)
        elif "_layernorm" in n and "weight" in n:
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def _starcoder_params_for_scheduler(model, weight_decay):
    """
    We use this to configure weight decay for StarCoder models.
    """
    params_with_wd = []
    params_without_wd = []
    for n, p in model.named_parameters():
        if "bias" in n:
            params_without_wd.append(p)
        elif "ln_" in n and "weight" in n:
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def _starcoder2_params_for_scheduler(model, weight_decay):
    """
    We use this to configure weight decay for StarCoder models.
    """
    params_with_wd = []
    params_without_wd = []
    for n, p in model.named_parameters():
        if "bias" in n:
            params_without_wd.append(p)
        elif "_layernorm" in n and "weight" in n:
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def _tarcoder_lora(model_path: str, lora_r, lora_alpha, lora_dropout):
    """
    Loads a StarCoder model for LoRA training.
    """

    from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_proj", "c_attn", "q_attn"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, use_cache=False, load_in_8bit=True
    ).cuda()
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
    return model


def get_params_for_scheduler(model: AutoModelForCausalLM, weight_decay: float = 0.001):
    arch = _get_architecture(model)
    if arch == "LlamaForCausalLM":
        return _llama3_params_for_scheduler(model, weight_decay)
    elif arch == "GPTBigCodeForCausalLM":
        return _starcoder_params_for_scheduler(model, weight_decay)
    elif arch == "Starcoder2ForCausalLM":
        return _starcoder2_params_for_scheduler(model, weight_decay)
    elif arch == "Qwen2ForCausalLM":
        return _qwen2_params_for_scheduler(model, weight_decay)
    elif arch == "Qwen3ForCausalLM":
        return _qwen3_params_for_scheduler(model, weight_decay)
    elif arch == "Olmo2ForCausalLM":
        return _olmo2_params_for_scheduler(model, weight_decay)
    elif arch == "Phi3ForCausalLM":
        return _phi3_params_for_scheduler(model, weight_decay)
    elif arch == "SmolLM3ForCausalLM":
        return _smollm3_params_for_scheduler(model, weight_decay)
    else:
        raise ValueError(f"Architecture {arch} not supported")