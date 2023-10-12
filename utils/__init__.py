import torch
import torchaudio
from torch import nn
from models import MaskNet

import yaml
import os
from prettytable import PrettyTable
import numpy as np

from typing import Tuple, Optional
from collections import OrderedDict


def generate_output(
        input_wav:torch.Tensor,
        model_path:str,
        model_config:str,
        device:torch.device=torch.device('cpu'),
        bandpass:bool=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    model = load_model(model_path, model_config, device)
    model.eval()
    input_wav = torch.unsqueeze(input_wav, dim=0)
    if bandpass:
        input_wav = bandpass_filter(input_wav, 0.0125, 0.25, 51)
    filtered_sound:torch.Tensor = model(input_wav)
    return filtered_sound[0, 0, :].detach(), filtered_sound[0, 1, :].detach()


def load_model(
        model_path:str, 
        model_config:str, 
        device:torch.device=torch.device('cpu')
    ) -> MaskNet:
    with open(model_config, 'r') as f:
        config = yaml.safe_load(f)
    model = MaskNet(**config)
    state_dict:OrderedDict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def save_model(model:MaskNet, pth:str) -> None:
    torch.save(model.state_dict(), pth)


def count_parameters(model:nn.Module, verbose:bool=False) -> int:
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    if verbose:
        print(table)
        print(f"Total Trainable Params: {total_params:,}")
    return total_params


def bandpass_filter(wav:torch.Tensor, c_low, c_high, filt_len):
    batch, channel, wav_len = wav.shape
    ir = sinc_impulse_response(torch.tensor(c_low), filt_len) - sinc_impulse_response(torch.tensor(c_high), filt_len)  # (filt_len)
    ir = torch.unsqueeze(ir, dim=0)    # (1, 1, filt_len)
    ir = ir.to(wav.device)
    wav_filt = nn.functional.conv1d(wav, ir, padding='same')
    return wav_filt


def sinc_impulse_response(cutoff: torch.Tensor, window_size: int = 513, high_pass: bool = False):
    if window_size % 2 == 0:
        raise ValueError(f"`window_size` must be odd. Given: {window_size}")

    half = window_size // 2
    device, dtype = cutoff.device, cutoff.dtype
    idx = torch.linspace(-half, half, window_size, device=device, dtype=dtype)

    filt = torch.special.sinc(cutoff.unsqueeze(-1) * idx.unsqueeze(0))
    filt = filt * torch.hamming_window(window_size, device=device, dtype=dtype, periodic=False).unsqueeze(0)
    filt = filt / filt.sum(dim=-1, keepdim=True).abs()

    if high_pass:
        filt = -filt
        filt[..., half] = 1.0 + filt[..., half]
    return filt
