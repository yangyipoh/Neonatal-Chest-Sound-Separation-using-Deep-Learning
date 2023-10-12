import torch
from torch import nn
from typing import Optional
from torchmetrics.functional.audio.sdr import (source_aggregated_signal_distortion_ratio, 
                                               signal_distortion_ratio, 
                                               scale_invariant_signal_distortion_ratio)


def normalise(x:torch.Tensor) -> torch.Tensor:
    """Normalise X such that sum(x) = 1

    Args:
        x (torch.Tensor): input tensor

    Returns:
        torch.Tensor: a tensor rescaled such that sum(x) = number of sources
    """
    return x/torch.sum(x)*len(x)


class SDR_Loss(nn.Module):
    """Creates a criterion that calculates the negative of SDR and the SI-SDR in a weighted manner

    Args:
        alpha (float): Weighting between SDR and SI-SDR. 
        alpha=1 would mean that the loss is contributed solely by SI-SDR.
        alpha=0 would mean that the loss is contributed solely by SDR.
        alpha=0.5 would mean equal contribution between SI-SDR and SDR. 
        Defaults to 0.5.
    """
    def __init__(self, weights:Optional[torch.Tensor]=None):
        super().__init__()
        if weights is None:
            weights = torch.tensor([1, 1, 0.1])
        self.weights = normalise(weights)

    def forward(self, output:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """Calculates the SDR loss, defined as
        L = -(1-alpha)*SDR - (alpha)*SI-SDR

        Args:
            output (torch.Tensor): Output tensor from the model (B, S, T)
            target (torch.Tensor): Target tensor (B, S, T)

        Returns:
            torch.Tensor: value of the loss function
        """
        sdr = scale_invariant_signal_distortion_ratio(preds=output, target=target, zero_mean=True)
        sdr = torch.mean(sdr, dim=0)
        sdr = torch.mean(sdr*self.weights.to(sdr.device))
        return -sdr
        

class LogMSE_Loss(nn.Module):
    """Creates a criterion that calculates the MSE that's weighted differently for different sources

    Args:
        weights (torch.Tensor): Weights for the different sources. Defaults to torch.tensor([1, 1, 0.1]).
    """
    def __init__(self, weights:Optional[torch.Tensor]=None):
        super().__init__()
        if weights is None:
            weights = torch.tensor([1, 1, 0.1])
        self.weights = normalise(weights)

    def forward(self, output:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """Calculates the weighted MSE loss

        Args:
            output (torch.Tensor): Output tensor form the model (B, S, T)
            target (torch.Tensor): Target tensor (B, S, T)

        Returns:
            torch.Tensor: value of the loss function
        """
        loss = (output - target)**2     # (B, S, T)
        loss = torch.mean(loss, dim=-1) # (B, S)
        loss = torch.mean(self.weights.to(loss.device)*loss)    # (B,)
        loss = torch.log10(loss+1)
        loss = torch.mean(loss)
        return loss
    

class SASDR_Loss(nn.Module):
    def __init__(self, weights:Optional[torch.Tensor]=None):
        super().__init__()
        # if weights is None:
        #     weights = torch.tensor([1, 1, 0.1])
        # self.weights = normalise(weights)

    def forward(self, output:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        snr = source_aggregated_signal_distortion_ratio(preds=output, target=target)
        snr = torch.mean(snr, dim=0)
        return -snr


class SNR_Loss(nn.Module):
    def __init__(self, weights:Optional[torch.Tensor]=None):
        super().__init__()
        if weights is None:
            weights = torch.tensor([1, 1, 0.1])
        self.weights = normalise(weights)

    def forward(self, output:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        snr = scale_invariant_signal_distortion_ratio(preds=output, target=target, zero_mean=False)
        snr = torch.mean(snr, dim=0)
        snr = torch.mean(snr*self.weights.to(snr.device))
        return -snr