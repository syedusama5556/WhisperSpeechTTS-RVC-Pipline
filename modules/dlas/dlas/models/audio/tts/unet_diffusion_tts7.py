import functools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from x_transformers import ContinuousTransformerWrapper, Encoder

import dlas.torch_intermediary as ml
from dlas.models.audio.tts.mini_encoder import AudioMiniEncoder
from dlas.models.diffusion.nn import (conv_nd, linear, normalization,
                                      timestep_embedding, zero_module)
from dlas.models.diffusion.unet_diffusion import (AttentionBlock, Downsample,
                                                  TimestepBlock,
                                                  TimestepEmbedSequential,
                                                  Upsample)
from dlas.scripts.audio.gen.use_diffuse_tts import ceil_multiple
from dlas.trainer.networks import register_model
from dlas.utils.util import checkpoint


def clustered_mask(probability, shape, dev, lateral_expansion_radius_max=3, inverted=False):
    """
    Produces a masking vector of the specified shape where each element has probability to be zero.
    lateral_expansion_radius_max neighbors of any element that is zero also have a 50% chance to be zero.
    Effectively, this produces clusters of masks tending to be lateral_expansion_radius_max wide.
    """
    # Each masked token spreads out to 1+lateral_expansion_radius_max on average, therefore reduce the probability in
    # kind
    probability = probability / (1+lateral_expansion_radius_max)

    mask = torch.rand(shape, device=dev)
    mask = (mask < probability).float()
    kernel = torch.tensor([.5 for _ in range(lateral_expansion_radius_max)] + [1] + [.5 for _ in range(lateral_expansion_radius_max)], device=dev)
    mask = F.conv1d(mask.unsqueeze(1), kernel.view(1,1,2*lateral_expansion_radius_max+1), padding=lateral_expansion_radius_max).squeeze(1)
    if inverted:
        return torch.bernoulli(torch.clamp(mask, 0, 1)) != 0
    else:
        return torch.bernoulli(torch.clamp(mask, 0, 1)) == 0


class CheckpointedLayer(nn.Module):
    """
    Wraps a module. When forward() is called, passes kwargs that require_grad through torch.checkpoint() and bypasses
    checkpoint for all other args.
    """
    def __init__(self, wrap):
        super().__init__()
        self.wrap = wrap

    def forward(self, x, *args, **kwargs):
        for k, v in kwargs.items():
            assert not (isinstance(v, torch.Tensor) and v.requires_grad)  # This would screw up checkpointing.
        partial = functools.partial(self.wrap, **kwargs)
        return torch.utils.checkpoint.checkpoint(partial, x, *args)


class CheckpointedXTransformerEncoder(nn.Module):
    """
    Wraps a ContinuousTransformerWrapper and applies CheckpointedLayer to each layer and permutes from channels-mid
    to channels-last that XTransformer expects.
    """
    def __init__(self, needs_permute=True, exit_permute=True, checkpoint=True, **xtransformer_kwargs):
        super().__init__()
        self.transformer = ContinuousTransformerWrapper(**xtransformer_kwargs)
        self.needs_permute = needs_permute
        self.exit_permute = exit_permute

        if not checkpoint:
            return
        for i in range(len(self.transformer.attn_layers.layers)):
            n, b, r = self.transformer.attn_layers.layers[i]
            self.transformer.attn_layers.layers[i] = nn.ModuleList([n, CheckpointedLayer(b), r])

    def forward(self, x, **kwargs):
        if self.needs_permute:
            x = x.permute(0,2,1)
        h = self.transformer(x, **kwargs)
        if self.exit_permute:
            h = h.permute(0,2,1)
        return h


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        padding = 1 if kernel_size == 3 else 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 1, padding=0),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

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
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class DiffusionTts(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    Customized to be conditioned on an aligned token prior.

    :param in_channels: channels in the input Tensor.
    :param num_tokens: number of tokens (e.g. characters) which can be provided.
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
            num_tokens=32,
            out_channels=2,  # mean and variance
            dropout=0,
            # res           1, 2, 4, 8,16,32,64,128,256,512, 1K, 2K
            channel_mult=  (1,1.5,2, 3, 4, 6, 8, 12, 16, 24, 32, 48),
            num_res_blocks=(1, 1, 1, 1, 1, 2, 2, 2,   2,  2,  2,  2),
            # spec_cond:    1, 0, 0, 1, 0, 0, 1, 0,   0,  1,  0,  0)
            # attn:         0, 0, 0, 0, 0, 0, 0, 0,   0,  1,  1,  1
            token_conditioning_resolutions=(1,16,),
            attention_resolutions=(512,1024,2048),
            conv_resample=True,
            dims=1,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            kernel_size=3,
            scale_factor=2,
            time_embed_dim_multiplier=4,
            cond_transformer_depth=8,
            mid_transformer_depth=8,
            # Parameters for regularization.
            nil_guidance_fwd_proportion=.3,
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
            # Parameters for super-sampling.
            super_sampling=False,
            super_sampling_max_noising_factor=.1,
            # Parameters for unaligned inputs.
            enabled_unaligned_inputs=False,
            num_unaligned_tokens=164,
            unaligned_encoder_depth=8,
            # Experimental parameters
            component_gradient_boosting=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if super_sampling:
            in_channels *= 2  # In super-sampling mode, the LR input is concatenated directly onto the input.
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
        self.nil_guidance_fwd_proportion = nil_guidance_fwd_proportion
        self.mask_token_id = num_tokens
        self.super_sampling_enabled = super_sampling
        self.super_sampling_max_noising_factor = super_sampling_max_noising_factor
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.component_gradient_boosting = component_gradient_boosting
        padding = 1 if kernel_size == 3 else 2

        time_embed_dim = model_channels * time_embed_dim_multiplier
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        embedding_dim = model_channels * 8
        # nn.Embedding
        self.code_embedding = ml.Embedding(num_tokens+1, embedding_dim)
        self.contextual_embedder = AudioMiniEncoder(1, embedding_dim, base_channels=32, depth=6, resnet_blocks=1,
                         attn_blocks=2, num_attn_heads=2, dropout=dropout, downsample_factor=4, kernel_size=5)
        self.conditioning_conv = nn.Conv1d(embedding_dim*3, embedding_dim, 1)

        self.enable_unaligned_inputs = enabled_unaligned_inputs
        if enabled_unaligned_inputs:
            # nn.Embedding
            self.unaligned_embedder = ml.Embedding(num_unaligned_tokens, embedding_dim)
            self.unaligned_encoder = CheckpointedXTransformerEncoder(
                max_seq_len=-1,
                use_pos_emb=False,
                attn_layers=Encoder(
                    dim=embedding_dim,
                    depth=unaligned_encoder_depth,
                    heads=num_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_emb_dim=True,
                )
            )

        self.conditioning_encoder = CheckpointedXTransformerEncoder(
                max_seq_len=-1,  # Should be unused
                use_pos_emb=False,
                attn_layers=Encoder(
                    dim=embedding_dim,
                    depth=cond_transformer_depth,
                    heads=num_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                    cross_attend=self.enable_unaligned_inputs,
                )
        )
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,embedding_dim,1))

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, kernel_size, padding=padding)
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
                token_conditioning_block = nn.Conv1d(embedding_dim, ch, 1)
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
                            ch, conv_resample, dims=dims, out_channels=out_ch, factor=scale_factor, ksize=1, pad=0
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        mid_transformer = CheckpointedXTransformerEncoder(
                max_seq_len=-1,  # Should be unused
                use_pos_emb=False,
                attn_layers=Encoder(
                    dim=ch,
                    depth=mid_transformer_depth,
                    heads=num_heads,
                    ff_dropout=dropout,
                    attn_dropout=dropout,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                )
            )
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                kernel_size=kernel_size,
            ),
            mid_transformer,
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                kernel_size=kernel_size,
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
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, factor=scale_factor)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, kernel_size, padding=padding)),
        )

    def get_grad_norm_parameter_groups(self):
        groups = {
            'minicoder': list(self.contextual_embedder.parameters()),
            'input_blocks': list(self.input_blocks.parameters()),
            'output_blocks': list(self.output_blocks.parameters()),
            'middle_transformer': list(self.middle_block.parameters()),
            'conditioning_encoder': list(self.conditioning_encoder.parameters())
        }
        if self.enable_unaligned_inputs:
            groups['unaligned_encoder'] = list(self.unaligned_encoder.parameters())
        return groups

    def before_step(self, it):
        if not self.component_gradient_boosting:
            return
        MIN_PROPORTIONAL_BOOST_LEVEL = .5
        MAX_MULTIPLIER = 100
        components = [list(self.contextual_embedder.parameters()), list(self.middle_block.parameters()), list(self.conditioning_encoder.parameters()),
                      list(self.unaligned_encoder.parameters())]
        input_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in self.input_blocks.parameters()]), 2)
        output_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in self.output_blocks.parameters()]), 2)
        diffusion_norm = (input_norm + output_norm) / 2
        min_norm = diffusion_norm * MIN_PROPORTIONAL_BOOST_LEVEL
        for component in components:
            norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in component]), 2)
            if norm < min_norm:
                mult = min_norm / (norm + 1e-8)
                mult = min(mult, MAX_MULTIPLIER)
                for p in component:
                    p.grad.data.mul_(mult)

    def forward(self, x, timesteps, tokens=None, conditioning_input=None, lr_input=None, unaligned_input=None, conditioning_free=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param tokens: an aligned text input.
        :param conditioning_input: a full-resolution audio clip that is used as a reference to the style you want decoded.
        :param lr_input: for super-sampling models, a guidance audio clip at a lower sampling rate.
        :param unaligned_input: A structural input that is not properly aligned with the output of the diffusion model.
                                Can be combined with a conditioning input to produce more robust conditioning.
        :param conditioning_free: When set, all conditioning inputs (including tokens, conditioning_input and unaligned_input) will not be considered.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert conditioning_input is not None
        if self.super_sampling_enabled:
            assert lr_input is not None
            if self.training and self.super_sampling_max_noising_factor > 0:
                noising_factor = random.uniform(0,self.super_sampling_max_noising_factor)
                lr_input = torch.randn_like(lr_input) * noising_factor + lr_input
            lr_input = F.interpolate(lr_input, size=(x.shape[-1],), mode='nearest')
            x = torch.cat([x, lr_input], dim=1)

        with autocast(x.device.type, enabled=self.enable_fp16):
            orig_x_shape = x.shape[-1]
            cm = ceil_multiple(x.shape[-1], 2048)
            if cm != 0:
                pc = (cm-x.shape[-1])/x.shape[-1]
                x = F.pad(x, (0,cm-x.shape[-1]))
                if tokens is not None:
                    tokens = F.pad(tokens, (0,int(pc*tokens.shape[-1])))

            hs = []
            time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

            if conditioning_free:
                code_emb = self.unconditioned_embedding.repeat(x.shape[0], 1, 1)
            else:
                if self.enable_unaligned_inputs:
                    assert unaligned_input is not None
                    unaligned_h = self.unaligned_embedder(unaligned_input).permute(0,2,1)
                    unaligned_h = self.unaligned_encoder(unaligned_h).permute(0,2,1)

                cond_emb = self.contextual_embedder(conditioning_input)
                if tokens is not None:
                    # Mask out guidance tokens for un-guided diffusion.
                    if self.training and self.nil_guidance_fwd_proportion > 0:
                        token_mask = clustered_mask(self.nil_guidance_fwd_proportion, tokens.shape, tokens.device, inverted=True)
                        tokens = torch.where(token_mask, self.mask_token_id, tokens)
                    code_emb = self.code_embedding(tokens).permute(0,2,1)
                    cond_emb = cond_emb.unsqueeze(-1).repeat(1,1,code_emb.shape[-1])
                    cond_time_emb = timestep_embedding(torch.zeros_like(timesteps), code_emb.shape[1])  # This was something I was doing (adding timesteps into this computation), but removed on second thought. TODO: completely remove.
                    cond_time_emb = cond_time_emb.unsqueeze(-1).repeat(1,1,code_emb.shape[-1])
                    code_emb = self.conditioning_conv(torch.cat([cond_emb, code_emb, cond_time_emb], dim=1))
                else:
                    code_emb = cond_emb.unsqueeze(-1)
                if self.enable_unaligned_inputs:
                    code_emb = self.conditioning_encoder(code_emb, context=unaligned_h)
                else:
                    code_emb = self.conditioning_encoder(code_emb)

            # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
            if self.training and self.unconditioned_percentage > 0:
                unconditioned_batches = torch.rand((code_emb.shape[0],1,1), device=code_emb.device) < self.unconditioned_percentage
                code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(x.shape[0], 1, 1), code_emb)

            first = True
            time_emb = time_emb.float()
            h = x
            for k, module in enumerate(self.input_blocks):
                if isinstance(module, nn.Conv1d):
                    h_tok = F.interpolate(module(code_emb), size=(h.shape[-1]), mode='nearest')
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
        return out[:, :, :orig_x_shape]


@register_model
def register_diffusion_tts7(opt_net, opt):
    return DiffusionTts(**opt_net['kwargs'])


# Test for ~4 second audio clip at 22050Hz
if __name__ == '__main__':
    clip = torch.randn(2, 1, 32768)
    tok = torch.randint(0,30, (2,388))
    cond = torch.randn(2, 1, 44000)
    ts = torch.LongTensor([600, 600])
    lr = torch.randn(2,1,10000)
    un = torch.randint(0,120, (2,100))
    model = DiffusionTts(128,
                         channel_mult=[1,1.5,2, 3, 4, 6, 8],
                         num_res_blocks=[2, 2, 2, 2, 2, 2, 1],
                         token_conditioning_resolutions=[1,4,16,64],
                         attention_resolutions=[],
                         num_heads=8,
                         kernel_size=3,
                         scale_factor=2,
                         time_embed_dim_multiplier=4, super_sampling=False,
                         enabled_unaligned_inputs=True,
                         component_gradient_boosting=True)
    o = model(clip, ts, tok, cond, lr, un)
    o.sum().backward()
    model.before_step(0)
    torch.save(model.state_dict(), 'test_out.pth')
