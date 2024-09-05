from typing import Mapping, Any
import importlib

from torch import nn
import torch


def get_obj_from_str(string: str, reload: bool=False) -> object:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def frozen_module(module: nn.Module) -> None:
    module.eval()
    module.train = disabled_train
    for p in module.parameters():
        p.requires_grad = False


def load_state_dict(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)


    # factorize
    if state_dict['hyper_encoder.entropy_bottleneck._offset'].size() != torch.Size([0]):
        state_dict['hyper_encoder.entropy_bottleneck._offset'] = torch.tensor([], dtype=torch.int32)
    if state_dict['hyper_encoder.entropy_bottleneck._quantized_cdf'].size() != torch.Size([0]):
        state_dict['hyper_encoder.entropy_bottleneck._quantized_cdf'] = torch.tensor([], dtype=torch.int32)
    if state_dict['hyper_encoder.entropy_bottleneck._cdf_length'].size() != torch.Size([0]):
        state_dict['hyper_encoder.entropy_bottleneck._cdf_length'] = torch.tensor([], dtype=torch.int32)

    # SGM
    if state_dict['hyper_encoder.gaussian_conditional._offset'].size() != torch.Size([0]):
        state_dict['hyper_encoder.gaussian_conditional._offset'] = torch.tensor([], dtype=torch.int32)
    if state_dict['hyper_encoder.gaussian_conditional._quantized_cdf'].size() != torch.Size([0]):
        state_dict['hyper_encoder.gaussian_conditional._quantized_cdf'] = torch.tensor([], dtype=torch.int32)
    if state_dict['hyper_encoder.gaussian_conditional._cdf_length'].size() != torch.Size([0]):
        state_dict['hyper_encoder.gaussian_conditional._cdf_length'] = torch.tensor([], dtype=torch.int32)
    if state_dict['hyper_encoder.gaussian_conditional.scale_table'].size() != torch.Size([0]):
        state_dict['hyper_encoder.gaussian_conditional.scale_table'] = torch.tensor([], dtype=torch.int32)

        

    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        
    # 320 4 3 3 -> 320 8 3 3 
    state_dict['model.diffusion_model.input_blocks.0.0.weight'] = state_dict['model.diffusion_model.input_blocks.0.0.weight'][:,:8,:,:]

    model.load_state_dict(state_dict, strict=strict)

