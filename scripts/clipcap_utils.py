from transformers import GPT2LMHeadModel, AutoConfig
from collections import OrderedDict
from safetensors.torch import load_file
from types import SimpleNamespace

from scripts.modeling_gpt2_cocomix import GPT2CoCoMixLMHeadModel

def merge_dicts(base, override):
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            merge_dicts(base[key], value)  # recurse
        else:
            base[key] = value  # override or add
    return base

def dict_to_obj(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_obj(i) for i in d]
    else:
        return d

class ParserObject:
    def __init__(self):
        self.gpt2_type = None
        self.prefix_length = 10
        self.prefix_length_clip = 10
        self.prefix_only = True # whether to freeze gpt2 and train only mapping network
        self.prefix_concept_enable = True # whether to extract concept feature from prefix
        self.batch_size = 64
        self.num_layers = 8
        self.normalize_prefix = False

def get_base_lm(cfg, gpt2_type):
    """define base model"""

    # Get model config
    config = AutoConfig.from_pretrained(gpt2_type['gpt2_config'])

    # Due to parallel wrapping, the weight names are changed
    state_dict = load_file(gpt2_type['base_gpt2_path'])
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    # Due to parallel wrapping, the weight names are changed

    if cfg.vocab_size is not None:
        config.vocab_size = cfg.vocab_size
    if cfg.n_embd is not None:
            config.n_embd = cfg.n_embd
    if cfg.n_layer is not None:
        config.n_layer = cfg.n_layer
    if cfg.n_head is not None:
        config.n_head = cfg.n_head

    if cfg.mode == "cocomix":
        config._attn_implementation = "eager"
        base_lm = GPT2CoCoMixLMHeadModel(
            config, cfg.concept_dim, cfg.insert_layer_index, cfg.concept_num
        )
    else:  # just next token prediction
        config._attn_implementation = "sdpa"
        base_lm = GPT2LMHeadModel(config)

    # Load pretrained
    base_lm.load_state_dict(new_state_dict, strict=False)
    # Weight tying because gpt-2 lm_head share weight with wte
    base_lm.lm_head.weight = base_lm.transformer.wte.weight

    return base_lm