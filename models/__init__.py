import torch
from torch import nn
import torchaudio
import ptwt
from torch.distributions.normal import Normal

from models.transformer import TransformerEncoder
from models.transformer_relative import TransformerRelativeEncoder
from models.conformer import ConformerEncoder
from models.deep_encoder import DeepEncoder

from typing import Tuple, Optional


class MaskGenerator(nn.Module):
    """The Mask Generator Model. Assumes that the input dimension has size (B, F, M)

    Args:
        input_dim (int): input feature size, F
        num_sources (int): number of mask generated, S
        num_feats (int): input feature size for the conformer model
        num_heads (int): number of multi-headed attention. Defaults to 4.
        ffn_expand (int): feedforward dimension expandsion. Defaults to 2.
        num_layers (int): number of conformer layers. Defaults to 4.
        conv_size (int): convolution size in the conformer, has to be an odd integer value. Defaults to 31.
        dropout (int): dropout rate in the conformer. Defaults to 0.0.
    """
    def __init__(self, 
        input_dim:int,
        output_dim:int,
        num_sources:int, 
        num_feats:int, 
        num_heads:int=4, 
        ffn_expand:int=2, # change to expansion factor
        num_layers:int=4,
        dropout:int=0.0,
        msk_type:str='transformer',
        use_conv:bool=False,
        kernel_size:int=3,
        conv_layers:int=7,
        individual_mask:bool = False,
    ):  
        super().__init__()
        # save this for reshaping in forward pass
        self.output_dim = output_dim
        self.num_feats = num_feats
        self.num_sources = num_sources

        # reshape input tensor for the conformer model
        self.input_norm = nn.GroupNorm(num_groups=input_dim, num_channels=input_dim, eps=1e-8)

        self.input_conv = nn.Conv1d(in_channels=input_dim, out_channels=num_feats, kernel_size=1)
        if use_conv:
            self.input_conv = nn.Sequential(
                self.input_conv, 
                DeepEncoder(
                    in_channels=num_feats,
                    out_channels=num_feats,
                    kernel_size=kernel_size,
                    num_layers=conv_layers,
                ),
            )
            
        # mask generator
        self.is_individual_mask = individual_mask
        if individual_mask:
            self.mask_generator = nn.ModuleList([self.get_mask_type(msk_type, num_feats, num_layers, num_heads, ffn_expand, dropout) for _ in range(num_sources)])
            
        else:
            self.mask_generator = self.get_mask_type(msk_type, num_feats, num_layers, num_heads, ffn_expand, dropout)
        

        # reshape output tensor to be size output_dim
        self.output_activation = nn.GELU()
        if individual_mask:
            self.output_conv = nn.ModuleList([nn.Conv1d(in_channels=num_feats, out_channels=output_dim, kernel_size=1) for _ in range(num_sources)])  # change num_feats to input_dim
        else:
            self.output_conv = nn.Conv1d(in_channels=num_feats, out_channels=output_dim*num_sources, kernel_size=1)  # change num_feats to input_dim*num_sources

        # ensure that the output mask is positive (and also add non-linearity in case the final layer is convolution)
        self.mask_activation = nn.ReLU()

    def get_mask_type(self, msk_type:str, num_feats:int, num_layers:int, 
                      num_heads:int, ffn_expand:int, dropout:float) -> nn.Module:
        if msk_type == 'transformer':
            return TransformerEncoder(
                encoder_dim=num_feats,
                num_layers=num_layers,
                num_attention_heads=num_heads,
                ffn_expansion=ffn_expand,
                dropout=dropout,
            )
        elif msk_type == 'transformer_relative':
            return TransformerRelativeEncoder(
                encoder_dim=num_feats,
                num_layers=num_layers,
                num_attention_heads=num_heads,
                ffn_expansion=ffn_expand,
                dropout=dropout,
            )
        elif msk_type == 'conformer':
            return ConformerEncoder(
                encoder_dim=num_feats,
                num_layers=num_layers,
                num_attention_heads=num_heads,
                feed_forward_expansion_factor=ffn_expand,
                dropout_p=dropout,
            )
        else:
            raise ValueError('Unknown Mask Type')
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """model feedforward

        Args:
            x (torch.Tensor): 3D Tensor with shape [B, F, T]

        Returns:
            torch.Tensor: 4D Tensor with shape [B, S, F, T]
        """
        batch_size = x.shape[0]

        # normalise and reshape feature vector
        x = self.input_conv(self.input_norm(x))  # [B, num_feat, T]
        
        if self.is_individual_mask:
            x = torch.permute(x, (0, 2, 1))     # [B, T, num_feat]
            x = [mask(x).permute(0, 2, 1) for mask in self.mask_generator]
            x = torch.stack(x, dim=1)           # [B, S, num_feat, T]

            x = self.output_activation(x)
            x = [out_conv(x[:, i, :]) for i, out_conv in enumerate(self.output_conv)]   # [B, F, T]*num_source
            return torch.stack(x, dim=1)
            
        else:
            x = torch.permute(x, (0, 2, 1))     # [B, T, num_feat]
            x = self.mask_generator(x)
            x = torch.permute(x, (0, 2, 1))   # [B, num_feat, T]

            x = self.output_activation(x)  
            x = self.output_conv(x)   # [B, F*S, T]
            x = self.mask_activation(x)

            return x.view(batch_size, self.num_sources, self.output_dim, -1) # [B, S, F, T]


class MaskNet(nn.Module):
    """MaskNet Model

    Args:
        num_sources (int): number of output source created. Defaults to 2.
        enc_kernel_size (int): kernel size of the encoder. Defaults to 16.
        enc_num_feats (int): number of channel created by encoder. Defaults to 512.
        enc_type (str): type of encoder used. Defaults to 'convolution'.
        dec_type (str): type of decoder used. Defaults to 'convolution'.
        msk_num_feats (int): number of features used in conformer model. Defaults to 256.
        msk_num_heads (int): number of multi-headed attention. Defaults to 4.
        msk_ffn_expand (int): feedforward dimension expandsion. Defaults to 2.
        msk_num_layers (int): number of conformer used. Defaults to 4.
        msk_conv_size (int): convolution kernel size in conformer. Defaults to 31.
        msk_dropout (float): dropout rate in conformer. Defaults to 0.0.

    Raises:
        Exception: enc_type must be in the list ["convolution", "spectrogram", "spectrogram2"]
        Exception: dec_type must be in the list ["convolution", "spectrogram", "group-convolution"]
        Assert: if dec_type is "spectrogram", enc_type must be "spectrogram"
    """
    def __init__(
        self,
        num_sources: int = 2,
        stochastic:bool = False,
        # encoder/decoder parameters
        enc_kernel_size: int = 512,
        enc_num_feats: int = 512,
        enc_type:str = 'convolution',
        dec_type:str = 'convolution',
        # mask generator parameters
        msk_num_feats: int = 256,
        msk_num_heads: int = 4,
        msk_ffn_expand: int = 2,
        msk_num_layers: int = 4,
        msk_use_conv:bool = False,
        msk_kernel_size:int = 3,
        msk_conv_layers:int = 3,
        msk_dropout: float = 0.0,
        msk_type: str = 'transformer',
        msk_individual_mask: bool = False,
        # wavelets
        use_wavelet: bool = False,
        wavelet_scale: int = 8,
        mother_wavelet: str = 'db4',
    ):
        super().__init__()

        # save some information about the model
        self.num_sources = num_sources
        self.is_stochastic = stochastic
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2

        # ----------------- encoder ---------------------
        self.enc_type = enc_type
        self.use_wavelet = use_wavelet
        self.mother_wavelet = mother_wavelet
        self.wavelet_scale = wavelet_scale
        self.encoder = self.get_encoder()
        
        # -------------- mask generator -----------------
        # double the input dimension to include phase information
        if self.enc_type == 'spectrogram2':
            mskgen_input_dim = enc_num_feats*2
        else:
            mskgen_input_dim = enc_num_feats
        self.mask_generator = MaskGenerator(
            input_dim=mskgen_input_dim,
            output_dim=mskgen_input_dim,
            num_sources=num_sources,
            # conformer stuff 
            num_feats=msk_num_feats,
            num_heads=msk_num_heads,
            ffn_expand=msk_ffn_expand,
            num_layers=msk_num_layers,
            dropout=msk_dropout,
            msk_type=msk_type,
            use_conv=msk_use_conv,
            kernel_size=msk_kernel_size,
            conv_layers=msk_conv_layers,
            individual_mask=msk_individual_mask,
        )

        # ---------------- decoder ---------------------
        self.dec_type = dec_type
        self.decoder = self.get_decoder()

        # if model output is stochastic, double the number of variables
        self.decoder_var = self.get_decoder() if stochastic else None

    def get_encoder(self) -> nn.Module:
        """generate model encoder

        Raises:
            Exception: self.enc_type must be in ["convolution", "spectrogram", "spectrogram2"]
            Exception: 

        Returns:
            nn.Module: encoder
        """
        if self.enc_type == 'convolution':
            encoder = nn.Conv1d(
                in_channels=self.wavelet_scale if self.use_wavelet else 1,
                out_channels=self.enc_num_feats,
                kernel_size=self.enc_kernel_size,
                stride=self.enc_stride,
                padding=self.enc_stride,
                bias=False,
            )
        elif self.enc_type == 'spectrogram':
            encoder = torchaudio.transforms.Spectrogram(
                n_fft=2*(self.enc_num_feats-1),
                win_length=self.enc_kernel_size,
                power=None,
            )
        else:
            raise Exception('Unknown encoder type')
        return encoder

    def get_decoder(self) -> nn.Module:
        """generate model decoder

        Raises:
            Exception: self.dec_type must be in ["convolution", "spectrogram", "group-convolution"]
            Exception: if self.dec_type is "spectrogram", self.enc_type must be in ["spectrogram", "spectrogram2"]

        Returns:
            nn.Module: decoder
        """
        if self.dec_type == 'convolution':
            decoder = nn.ConvTranspose1d(
                in_channels=self.enc_num_feats,
                out_channels=1,
                kernel_size=self.enc_kernel_size,
                stride=self.enc_stride,
                padding=self.enc_stride,
                bias=False,
            )
        elif self.dec_type == 'group-convolution':
            decoder = nn.ModuleList([nn.ConvTranspose1d(
                in_channels=self.enc_num_feats,
                out_channels=1,
                kernel_size=self.enc_kernel_size,
                stride=self.enc_stride,
                padding=self.enc_stride,
                bias=False,
            ) for _ in range(self.num_sources)])
        elif self.dec_type == 'spectrogram':
            assert self.enc_type == 'spectrogram', 'Input has to be a spectrogram if decoder is a spectrogram'
            assert not self.is_stochastic, 'Spectrogram cannot be stochastic'
            decoder = torchaudio.transforms.InverseSpectrogram(
                n_fft=2*(self.enc_num_feats-1),
                win_length=self.enc_kernel_size,
            )
        else:
            raise Exception('Unknown decoder type')
        
        return decoder


    def _align_num_frames_with_strides(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad input sequence such that reconstruction can be done

        Args:
            input (torch.Tensor): input sequence

        Returns:
            Tuple[torch.Tensor, int]: padded input sequence and the amount of padding
        """
        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return input, 0

        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(
            batch_size,
            num_channels,
            num_paddings,
            dtype=input.dtype,
            device=input.device,
        )
        return torch.cat([input, pad], 2), num_paddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """model feedforward
        B: batch size
        L: input frame length
        L': padded input frame length
        F: feature dimension
        M: feature frame length
        S: number of sources

        Flowchart of the model
        input --> encoder --> MaskGenerator --> decoder

        Args:
            x (torch.Tensor): input sequence

        Raises:
            Exception: input must be of shape (B, channel==1, T)

        Returns:
            torch.Tensor: separated sequence of shape (B, S, T)
        """
        if x.ndim != 3 or x.shape[1] != 1:
            raise Exception(f"Expected 3D tensor (B, channel==1, T). Found: {x.shape}")

        # preprocessing
        input_frame_length = x.shape[2]     # save L for reconstruction later
        x, num_pads = self._align_num_frames_with_strides(x)  # B, 1, T'

        # encoder
        if self.use_wavelet:
            x = self.wavelet_analysis(x)
        x, phase = self.encoder_forward(x)     # B, F, M

        # mask generator
        x = self.mask_generator(x) * x.unsqueeze(1)  # B, S, F, M

        # decoder
        if self.is_stochastic:
            var = self.decoder_forward(x, phase, num_pads, input_frame_length, is_var=True)
        x = self.decoder_forward(x, phase, num_pads, input_frame_length, is_var=False)       # B, S, T'
        
        if not self.is_stochastic:
            return x
        m = Normal(loc=x, scale=var)
        x = m.rsample()
        return x
    
    def wavelet_analysis(self, x:torch.Tensor):
        batch, _, frame = x.shape
        coeffs = ptwt.wavedec(x, wavelet=self.mother_wavelet, level=self.wavelet_scale)
        cfd = torch.zeros(batch, self.wavelet_scale, frame).to(x.device)
        for k, coeff in enumerate(coeffs[:1:-1]):
            d = coeff.repeat_interleave(2**(k+1), dim=2)
            diff = d.shape[2] - frame
            cfd[:, k, :] = d[:, 0, diff//2:d.shape[2]-diff//2]
        return cfd
    
    def encoder_forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """encoder feedforward

        Args:
            x (torch.Tensor): input seqeuence of size (B, 1, L')

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: output feature (B, F, M) and phase (if applicable), (B, F, M)
        """
        if self.enc_type == 'convolution':
            x = self.encoder(x)         # B, F, M
            return x, None
        elif self.enc_type == 'spectrogram':
            x = self.encoder(x.squeeze(dim=1))  # B, F, M
            phase = torch.angle(x)      # B, F, M
            x = torch.abs(x)            # B, F, M
            return x, phase
        
    def decoder_forward(self, 
        x:torch.Tensor, 
        phase:Optional[torch.Tensor], 
        num_pads:int, 
        input_frame_length:int,
        is_var:bool,
    ) -> torch.Tensor:
        """decoder feedforward

        Args:
            x (torch.Tensor): masked input
            phase (torch.Tensor): phase input (used in spectrogram)
            num_pads (int): number of padding used (used in convolution and group-convolution)
            input_frame_length (int): input frame length, T

        Returns:
            torch.Tensor: separated sources mean or var of shape (B, S, T)
        """
        # pick decoder
        if is_var:
            decoder = self.decoder_var
        else:
            decoder = self.decoder
        
        if self.dec_type == 'convolution':
            batch_size = x.shape[0]
            x = x.view(batch_size * self.num_sources, self.enc_num_feats, -1)  # B*S, F, M
            x = decoder(x)  # B*S, 1, L'
            x = x.view(batch_size, self.num_sources, -1)  # B, S, L'
            if num_pads > 0:
                x = x[..., :-num_pads]  # B, S, L
        elif self.dec_type == 'group-convolution':
            x = [decoder_single(x[:, i, :]) for i, decoder_single in enumerate(decoder)]
            x = torch.cat(x, 1)     # B, S, T'
            if num_pads > 0:
                x = x[..., :-num_pads]  # B, S, L
        elif self.dec_type == 'spectrogram':
            x = torch.polar(x, phase.unsqueeze(1)) if self.enc_type == 'spectrogram' else torch.polar(x, phase)  # B, S, F, M
            x = decoder(x, input_frame_length)  # B, S, L

        if is_var:
            x = nn.functional.relu(x) + 1.e-7
        return x