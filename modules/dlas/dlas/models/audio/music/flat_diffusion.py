import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

import dlas.torch_intermediary as ml
from dlas.models.arch_util import ResBlock
from dlas.models.diffusion.nn import (conv_nd, linear, normalization,
                                      timestep_embedding, zero_module)
from dlas.models.diffusion.unet_diffusion import (AttentionBlock,
                                                  TimestepBlock,
                                                  TimestepEmbedSequential)
from dlas.trainer.networks import register_model
from dlas.utils.util import checkpoint


def is_latent(t):
    return t.dtype == torch.float


def is_sequence(t):
    return t.dtype == torch.long


class MultiGroupEmbedding(nn.Module):
    def __init__(self, tokens, groups, dim):
        super().__init__()
        # nn.Embedding
        self.m = nn.ModuleList(
            [ml.Embedding(tokens, dim // groups) for _ in range(groups)])

    def forward(self, x):
        h = [embedding(x[:, :, i]) for i, embedding in enumerate(self.m)]
        return torch.cat(h, dim=-1)


class TimestepResBlock(TimestepBlock):
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


class DiffusionLayer(TimestepBlock):
    def __init__(self, model_channels, dropout, num_heads):
        super().__init__()
        self.resblk = TimestepResBlock(
            model_channels, model_channels, dropout, model_channels, dims=1, use_scale_shift_norm=True)
        self.attn = AttentionBlock(
            model_channels, num_heads, relative_pos_embeddings=True)

    def forward(self, x, time_emb):
        y = self.resblk(x, time_emb)
        return self.attn(y)


class FlatDiffusion(nn.Module):
    def __init__(
            self,
            model_channels=512,
            num_layers=8,
            in_channels=256,
            in_latent_channels=512,
            token_count=8,
            in_groups=None,
            out_channels=512,  # mean and variance
            dropout=0,
            use_fp16=False,
            num_heads=8,
            # Parameters for regularization.
            layer_drop=.1,
            # This implements a mechanism similar to what is used in classifier-free training.
            unconditioned_percentage=.1,
            train_mel_head=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_heads = num_heads
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.layer_drop = layer_drop

        self.inp_block = conv_nd(1, in_channels, model_channels, 3, 1, 1)
        self.time_embed = nn.Sequential(
            linear(model_channels, model_channels),
            nn.SiLU(),
            linear(model_channels, model_channels),
        )

        # Either code_converter or latent_converter is used, depending on what type of conditioning data is fed.
        # This model is meant to be able to be trained on both for efficiency purposes - it is far less computationally
        # complex to generate tokens, while generating latents will normally mean propagating through a deep autoregressive
        # transformer network.
        if in_groups is None:
            # nn.Embedding
            self.embeddings = ml.Embedding(token_count, model_channels)
        else:
            self.embeddings = MultiGroupEmbedding(
                token_count, in_groups, model_channels)
        self.latent_conditioner = nn.Sequential(
            nn.Conv1d(in_latent_channels, model_channels, 3, padding=1),
            AttentionBlock(model_channels, num_heads,
                           relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads,
                           relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads,
                           relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads,
                           relative_pos_embeddings=True),
        )
        self.code_converter = nn.Sequential(
            ResBlock(dims=1, channels=model_channels, dropout=dropout),
            AttentionBlock(model_channels, num_heads,
                           relative_pos_embeddings=True),
            ResBlock(dims=1, channels=model_channels, dropout=dropout),
            AttentionBlock(model_channels, num_heads,
                           relative_pos_embeddings=True),
            ResBlock(dims=1, channels=model_channels, dropout=dropout),
        )
        self.code_norm = normalization(model_channels)
        self.contextual_embedder = nn.Sequential(nn.Conv1d(in_channels, model_channels, 3, padding=1, stride=2),
                                                 nn.Conv1d(
                                                     model_channels, model_channels*2, 3, padding=1, stride=2),
                                                 AttentionBlock(
                                                     model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(
                                                     model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(
                                                     model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(
                                                     model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False))
        self.unconditioned_embedding = nn.Parameter(
            torch.randn(1, model_channels, 1))
        self.conditioning_timestep_integrator = TimestepEmbedSequential(
            DiffusionLayer(model_channels, dropout, num_heads),
            DiffusionLayer(model_channels, dropout, num_heads),
            DiffusionLayer(model_channels, dropout, num_heads),
        )
        self.integrating_conv = nn.Conv1d(
            model_channels*2, model_channels, kernel_size=1)
        self.mel_head = nn.Conv1d(
            model_channels, in_channels, kernel_size=3, padding=1)

        self.layers = nn.ModuleList([DiffusionLayer(model_channels, dropout, num_heads) for _ in range(num_layers)] +
                                    [TimestepResBlock(model_channels, model_channels, dropout, dims=1, use_scale_shift_norm=True) for _ in range(3)])

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels,
                        out_channels, 3, padding=1)),
        )

        if train_mel_head:
            for m in [self.conditioning_timestep_integrator, self.integrating_conv, self.layers,
                      self.out]:
                for p in m.parameters():
                    p.requires_grad = False
                    p.DO_NOT_TRAIN = True

    def get_grad_norm_parameter_groups(self):
        groups = {
            'minicoder': list(self.contextual_embedder.parameters()),
            'layers': list(self.layers.parameters()),
            'code_converters': list(self.embeddings.parameters()) + list(self.code_converter.parameters()) + list(self.latent_conditioner.parameters()),
            'timestep_integrator': list(self.conditioning_timestep_integrator.parameters()) + list(self.integrating_conv.parameters()),
            'time_embed': list(self.time_embed.parameters()),
        }
        return groups

    def timestep_independent(self, aligned_conditioning, conditioning_input, expected_seq_len, return_code_pred):
        # Shuffle aligned_latent to BxCxS format
        if is_latent(aligned_conditioning):
            aligned_conditioning = aligned_conditioning.permute(0, 2, 1)

        # Note: this block does not need to repeated on inference, since it is not timestep-dependent or x-dependent.
        speech_conditioning_input = conditioning_input.unsqueeze(1) if len(
            conditioning_input.shape) == 3 else conditioning_input
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.contextual_embedder(
                speech_conditioning_input[:, j]))
        conds = torch.cat(conds, dim=-1)
        cond_emb = conds.mean(dim=-1)
        cond_scale, cond_shift = torch.chunk(cond_emb, 2, dim=1)
        if is_latent(aligned_conditioning):
            code_emb = self.latent_conditioner(aligned_conditioning)
        else:
            code_emb = self.embeddings(aligned_conditioning)
            code_emb = code_emb.permute(0, 2, 1)

        unconditioned_batches = torch.zeros(
            (code_emb.shape[0], 1, 1), device=code_emb.device)
        # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                               device=code_emb.device) < self.unconditioned_percentage
            code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(aligned_conditioning.shape[0], 1, 1),
                                   code_emb)
        expanded_code_emb = F.interpolate(
            code_emb, size=expected_seq_len, mode='nearest')
        expanded_code_emb = self.code_converter(expanded_code_emb)
        expanded_code_emb = self.code_norm(
            expanded_code_emb) * (1 + cond_scale.unsqueeze(-1)) + cond_shift.unsqueeze(-1)

        if not return_code_pred:
            return expanded_code_emb
        else:
            mel_pred = self.mel_head(expanded_code_emb)
            # Multiply mel_pred by !unconditioned_branches, which drops the gradient on unconditioned branches. This is because we don't want that gradient being used to train parameters through the codes_embedder as it unbalances contributions to that network from the MSE loss.
            mel_pred = mel_pred * unconditioned_batches.logical_not()
            return expanded_code_emb, mel_pred

    def forward(self, x, timesteps, codes=None, conditioning_input=None, precomputed_aligned_embeddings=None, conditioning_free=False, return_code_pred=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param codes: an aligned latent or sequence of tokens providing useful data about the sample to be produced.
        :param conditioning_input: a full-resolution audio clip that is used as a reference to the style you want decoded.
        :param precomputed_aligned_embeddings: Embeddings returned from self.timestep_independent()
        :param conditioning_free: When set, all conditioning inputs (including tokens and conditioning_input) will not be considered.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert precomputed_aligned_embeddings is not None or (
            codes is not None and conditioning_input is not None)
        # These two are mutually exclusive.
        assert not (
            return_code_pred and precomputed_aligned_embeddings is not None)

        unused_params = []
        if conditioning_free:
            code_emb = self.unconditioned_embedding.repeat(
                x.shape[0], 1, x.shape[-1])
            unused_params.extend(
                list(self.code_converter.parameters()) + list(self.code_embedding.parameters()))
            unused_params.extend(list(self.latent_conditioner.parameters()))
        else:
            if precomputed_aligned_embeddings is not None:
                code_emb = precomputed_aligned_embeddings
            else:
                code_emb, mel_pred = self.timestep_independent(
                    codes, conditioning_input, x.shape[-1], True)
                if is_latent(codes):
                    unused_params.extend(
                        list(self.code_converter.parameters()) + list(self.code_embedding.parameters()))
                else:
                    unused_params.extend(
                        list(self.latent_conditioner.parameters()))

            unused_params.append(self.unconditioned_embedding)

        time_emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels))
        code_emb = self.conditioning_timestep_integrator(code_emb, time_emb)
        x = self.inp_block(x)
        x = torch.cat([x, code_emb], dim=1)
        x = self.integrating_conv(x)
        for i, lyr in enumerate(self.layers):
            # Do layer drop where applicable. Do not drop first and last layers.
            if self.training and self.layer_drop > 0 and i != 0 and i != (len(self.layers)-1) and random.random() < self.layer_drop:
                unused_params.extend(list(lyr.parameters()))
            else:
                # First and last blocks will have autocast disabled for improved precision.
                with autocast(x.device.type, enabled=self.enable_fp16 and i != 0):
                    x = lyr(x, time_emb)

        x = x.float()
        out = self.out(x)

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        if return_code_pred:
            return out, mel_pred
        return out

    def get_conditioning_latent(self, conditioning_input):
        speech_conditioning_input = conditioning_input.unsqueeze(1) if len(
            conditioning_input.shape) == 3 else conditioning_input
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.contextual_embedder(
                speech_conditioning_input[:, j]))
        conds = torch.cat(conds, dim=-1)
        return conds.mean(dim=-1)


@register_model
def register_flat_diffusion(opt_net, opt):
    return FlatDiffusion(**opt_net['kwargs'])


if __name__ == '__main__':
    clip = torch.randn(2, 256, 400)
    aligned_latent = torch.randn(2, 388, 512)
    aligned_sequence = torch.randint(0, 8, (2, 100, 8))
    cond = torch.randn(2, 256, 400)
    ts = torch.LongTensor([600, 600])
    model = FlatDiffusion(
        512, layer_drop=.3, unconditioned_percentage=.5, in_groups=8, train_mel_head=True)
    # Test with latent aligned conditioning
    # o = model(clip, ts, aligned_latent, cond)
    # Test with sequence aligned conditioning
    o = model(clip, ts, aligned_sequence, cond, return_code_pred=True)
