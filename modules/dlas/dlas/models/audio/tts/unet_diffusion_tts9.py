import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from x_transformers import Encoder

import dlas.torch_intermediary as ml
from dlas.models.audio.tts.mini_encoder import AudioMiniEncoder
from dlas.models.audio.tts.unet_diffusion_tts7 import \
    CheckpointedXTransformerEncoder
from dlas.models.diffusion.nn import (conv_nd, linear, normalization,
                                      timestep_embedding, zero_module)
from dlas.models.diffusion.unet_diffusion import (AttentionBlock, Downsample,
                                                  TimestepBlock,
                                                  TimestepEmbedSequential,
                                                  Upsample)
from dlas.scripts.audio.gen.use_diffuse_tts import ceil_multiple
from dlas.trainer.networks import register_model
from dlas.utils.util import checkpoint


def is_latent(t):
    return t.dtype == torch.float


def is_sequence(t):
    return t.dtype == torch.long


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        kernel_size=3,
        efficient_config=True,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = {1: 0, 3: 1, 5: 2}[kernel_size]
        eff_kernel = 1 if efficient_config else 3
        eff_padding = 0 if efficient_config else 1

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels,
                    eff_kernel, padding=eff_padding),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels,
                        kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, eff_kernel, padding=eff_padding)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, x, emb
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class DiffusionTts(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    Customized to be conditioned on an aligned prior derived from a autoregressive
    GPT-style model.

    :param in_channels: channels in the input Tensor.
    :param in_latent_channels: channels from the input latent.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            model_channels,
            in_channels=1,
            in_latent_channels=1024,
            in_tokens=8193,
            conditioning_dim_factor=8,
            conditioning_expansion=4,
            out_channels=2,  # mean and variance
            dropout=0,
            # res           1, 2, 4, 8,16,32,64,128,256,512, 1K, 2K
            channel_mult=(1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48),
            num_res_blocks=(1, 1, 1, 1, 1, 2, 2, 2,   2,  2,  2,  2),
            # spec_cond:    1, 0, 0, 1, 0, 0, 1, 0,   0,  1,  0,  0)
            # attn:         0, 0, 0, 0, 0, 0, 0, 0,   0,  1,  1,  1
            token_conditioning_resolutions=(1, 16,),
            attention_resolutions=(512, 1024, 2048),
            conv_resample=True,
            dims=1,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            kernel_size=3,
            scale_factor=2,
            time_embed_dim_multiplier=4,
            freeze_main_net=False,
            # Uses kernels with width of 1 in several places rather than 3.
            efficient_convs=True,
            use_scale_shift_norm=True,
            # Parameters for regularization.
            # This implements a mechanism similar to what is used in classifier-free training.
            unconditioned_percentage=.1,
            # Parameters for super-sampling.
            super_sampling=False,
            super_sampling_max_noising_factor=.1,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if super_sampling:
            # In super-sampling mode, the LR input is concatenated directly onto the input.
            in_channels *= 2
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dims = dims
        self.super_sampling_enabled = super_sampling
        self.super_sampling_max_noising_factor = super_sampling_max_noising_factor
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.alignment_size = 2 ** (len(channel_mult)+1)
        self.freeze_main_net = freeze_main_net
        padding = 1 if kernel_size == 3 else 2
        down_kernel = 1 if efficient_convs else 3

        time_embed_dim = model_channels * time_embed_dim_multiplier
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        conditioning_dim = model_channels * conditioning_dim_factor
        # Either code_converter or latent_converter is used, depending on what type of conditioning data is fed.
        # This model is meant to be able to be trained on both for efficiency purposes - it is far less computationally
        # complex to generate tokens, while generating latents will normally mean propagating through a deep autoregressive
        # transformer network.
        self.code_converter = nn.Sequential(
            # nn.Embedding
            ml.Embedding(in_tokens, conditioning_dim),
            CheckpointedXTransformerEncoder(
                needs_permute=False,
                max_seq_len=-1,
                use_pos_emb=False,
                attn_layers=Encoder(
                    dim=conditioning_dim,
                    depth=3,
                    heads=num_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                )
            ))
        self.latent_converter = nn.Conv1d(
            in_latent_channels, conditioning_dim, 1)
        self.aligned_latent_padding_embedding = nn.Parameter(
            torch.randn(1, in_latent_channels, 1))
        if in_channels > 60:  # It's a spectrogram.
            self.contextual_embedder = nn.Sequential(nn.Conv1d(in_channels, conditioning_dim, 3, padding=1, stride=2),
                                                     CheckpointedXTransformerEncoder(
                                                         needs_permute=True,
                                                         max_seq_len=-1,
                                                         use_pos_emb=False,
                                                         attn_layers=Encoder(
                                                             dim=conditioning_dim,
                                                             depth=4,
                                                             heads=num_heads,
                                                             ff_dropout=dropout,
                                                             attn_dropout=dropout,
                                                             use_rmsnorm=True,
                                                             ff_glu=True,
                                                             rotary_pos_emb=True,
                                                         )
            ))
        else:
            self.contextual_embedder = AudioMiniEncoder(1, conditioning_dim, base_channels=32, depth=6, resnet_blocks=1,
                                                        attn_blocks=3, num_attn_heads=8, dropout=dropout, downsample_factor=4, kernel_size=5)
        self.conditioning_conv = nn.Conv1d(
            conditioning_dim*2, conditioning_dim, 1)
        self.unconditioned_embedding = nn.Parameter(
            torch.randn(1, conditioning_dim, 1))
        self.conditioning_timestep_integrator = TimestepEmbedSequential(
            ResBlock(conditioning_dim, time_embed_dim, dropout, out_channels=conditioning_dim,
                     dims=dims, kernel_size=1, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(conditioning_dim, num_heads=num_heads,
                           num_head_channels=num_head_channels),
            ResBlock(conditioning_dim, time_embed_dim, dropout, out_channels=conditioning_dim,
                     dims=dims, kernel_size=1, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(conditioning_dim, num_heads=num_heads,
                           num_head_channels=num_head_channels),
            ResBlock(conditioning_dim, time_embed_dim, dropout, out_channels=conditioning_dim,
                     dims=dims, kernel_size=1, use_scale_shift_norm=use_scale_shift_norm),
        )
        self.conditioning_expansion = conditioning_expansion

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels,
                            kernel_size, padding=padding)
                )
            ]
        )
        token_conditioning_blocks = []
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, (mult, num_blocks) in enumerate(zip(channel_mult, num_res_blocks)):
            if ds in token_conditioning_resolutions:
                token_conditioning_block = nn.Conv1d(conditioning_dim, ch, 1)
                token_conditioning_block.weight.data *= .02
                self.input_blocks.append(token_conditioning_block)
                token_conditioning_blocks.append(token_conditioning_block)

            for _ in range(num_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        kernel_size=kernel_size,
                        efficient_config=efficient_convs,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, factor=scale_factor, ksize=down_kernel, pad=0 if down_kernel == 1 else 1
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                kernel_size=kernel_size,
                efficient_config=efficient_convs,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                kernel_size=kernel_size,
                efficient_config=efficient_convs,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, (mult, num_blocks) in list(enumerate(zip(channel_mult, num_res_blocks)))[::-1]:
            for i in range(num_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        kernel_size=kernel_size,
                        efficient_config=efficient_convs,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                        )
                    )
                if level and i == num_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, conv_resample, dims=dims,
                                 out_channels=out_ch, factor=scale_factor)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels,
                        kernel_size, padding=padding)),
        )

        if self.freeze_main_net:
            mains = [self.time_embed, self.contextual_embedder, self.conditioning_conv, self.unconditioned_embedding, self.conditioning_timestep_integrator,
                     self.input_blocks, self.middle_block, self.output_blocks, self.out]
            for m in mains:
                for p in m.parameters():
                    p.requires_grad = False
                    p.DO_NOT_TRAIN = True

    def get_grad_norm_parameter_groups(self):
        if self.freeze_main_net:
            return {}
        groups = {
            'minicoder': list(self.contextual_embedder.parameters()),
            'input_blocks': list(self.input_blocks.parameters()),
            'output_blocks': list(self.output_blocks.parameters()),
            'middle_transformer': list(self.middle_block.parameters()),
        }
        return groups

    def fix_alignment(self, x, aligned_conditioning):
        """
        The UNet requires that the input <x> is a certain multiple of 2, defined by the UNet depth. Enforce this by
        padding both <x> and <aligned_conditioning> before forward propagation and removing the padding before returning.
        """
        cm = ceil_multiple(x.shape[-1], self.alignment_size)
        if cm != 0:
            pc = (cm-x.shape[-1])/x.shape[-1]
            x = F.pad(x, (0, cm-x.shape[-1]))
            # Also fix aligned_latent, which is aligned to x.
            if is_latent(aligned_conditioning):
                aligned_conditioning = torch.cat([aligned_conditioning,
                                                  self.aligned_latent_padding_embedding.repeat(x.shape[0], 1, int(pc * aligned_conditioning.shape[-1]))], dim=-1)
            else:
                aligned_conditioning = F.pad(
                    aligned_conditioning, (0, int(pc*aligned_conditioning.shape[-1])))
        return x, aligned_conditioning

    def forward(self, x, timesteps, aligned_conditioning, conditioning_input, lr_input=None, conditioning_free=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param aligned_conditioning: an aligned latent or sequence of tokens providing useful data about the sample to be produced.
        :param conditioning_input: a full-resolution audio clip that is used as a reference to the style you want decoded.
        :param lr_input: for super-sampling models, a guidance audio clip at a lower sampling rate.
        :param conditioning_free: When set, all conditioning inputs (including tokens and conditioning_input) will not be considered.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert conditioning_input is not None
        if self.super_sampling_enabled:
            assert lr_input is not None
            if self.training and self.super_sampling_max_noising_factor > 0:
                noising_factor = random.uniform(
                    0, self.super_sampling_max_noising_factor)
                lr_input = torch.randn_like(
                    lr_input) * noising_factor + lr_input
            lr_input = F.interpolate(
                lr_input, size=(x.shape[-1],), mode='nearest')
            x = torch.cat([x, lr_input], dim=1)

        # Shuffle aligned_latent to BxCxS format
        if is_latent(aligned_conditioning):
            aligned_conditioning = aligned_conditioning.permute(0, 2, 1)

        # Fix input size to the proper multiple of 2 so we don't get alignment errors going down and back up the U-net.
        orig_x_shape = x.shape[-1]
        x, aligned_conditioning = self.fix_alignment(x, aligned_conditioning)

        with autocast(x.device.type, enabled=self.enable_fp16):

            hs = []
            time_emb = self.time_embed(
                timestep_embedding(timesteps, self.model_channels))

            # Note: this block does not need to repeated on inference, since it is not timestep-dependent.
            if conditioning_free:
                code_emb = self.unconditioned_embedding.repeat(
                    x.shape[0], 1, 1)
            else:
                cond_emb = self.contextual_embedder(conditioning_input)
                if len(cond_emb.shape) == 3:  # Just take the first element.
                    cond_emb = cond_emb[:, :, 0]
                if is_latent(aligned_conditioning):
                    code_emb = self.latent_converter(aligned_conditioning)
                else:
                    code_emb = self.code_converter(aligned_conditioning)
                cond_emb = cond_emb.unsqueeze(-1).repeat(1,
                                                         1, code_emb.shape[-1])
                code_emb = self.conditioning_conv(
                    torch.cat([cond_emb, code_emb], dim=1))
            # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
            if self.training and self.unconditioned_percentage > 0:
                unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                                   device=code_emb.device) < self.unconditioned_percentage
                code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(x.shape[0], 1, 1),
                                       code_emb)

            # Everything after this comment is timestep dependent.
            code_emb = torch.repeat_interleave(
                code_emb, self.conditioning_expansion, dim=-1)
            code_emb = self.conditioning_timestep_integrator(
                code_emb, time_emb)

            first = True
            time_emb = time_emb.float()
            h = x
            for k, module in enumerate(self.input_blocks):
                if isinstance(module, nn.Conv1d):
                    h_tok = F.interpolate(module(code_emb), size=(
                        h.shape[-1]), mode='nearest')
                    h = h + h_tok
                else:
                    with autocast(x.device.type, enabled=self.enable_fp16 and not first):
                        # First block has autocast disabled to allow a high precision signal to be properly vectorized.
                        h = module(h, time_emb)
                    hs.append(h)
                first = False
            h = self.middle_block(h, time_emb)
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, time_emb)

        # Last block also has autocast disabled for high-precision outputs.
        h = h.float()
        out = self.out(h)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        params = [self.aligned_latent_padding_embedding,
                  self.unconditioned_embedding] + list(self.latent_converter.parameters())
        for p in params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        return out[:, :, :orig_x_shape]


@register_model
def register_diffusion_tts9(opt_net, opt):
    return DiffusionTts(**opt_net['kwargs'])


if __name__ == '__main__':
    clip = torch.randn(2, 1, 32868)
    aligned_latent = torch.randn(2, 388, 1024)
    aligned_sequence = torch.randint(0, 8192, (2, 388))
    cond = torch.randn(2, 1, 44000)
    ts = torch.LongTensor([600, 600])
    model = DiffusionTts(128,
                         channel_mult=[1, 1.5, 2, 3, 4, 6, 8],
                         num_res_blocks=[2, 2, 2, 2, 2, 2, 1],
                         token_conditioning_resolutions=[1, 4, 16, 64],
                         attention_resolutions=[],
                         num_heads=8,
                         kernel_size=3,
                         scale_factor=2,
                         time_embed_dim_multiplier=4,
                         super_sampling=False,
                         efficient_convs=False)
    # Test with latent aligned conditioning
    o = model(clip, ts, aligned_latent, cond)
    # Test with sequence aligned conditioning
    o = model(clip, ts, aligned_sequence, cond)
