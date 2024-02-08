import itertools
from random import randrange

import torch
import torch.nn as nn
import torch.nn.functional as F

import dlas.torch_intermediary as ml
from dlas.models.arch_util import (AttentionBlock, RelativeQKBias, ResBlock,
                                   TimestepEmbedSequential,
                                   build_local_attention_mask, cGLU)
from dlas.models.diffusion.nn import (conv_nd, linear, normalization,
                                      timestep_embedding, zero_module)
from dlas.models.diffusion.unet_diffusion import TimestepBlock
from dlas.trainer.networks import register_model
from dlas.utils.util import checkpoint


class SubBlock(nn.Module):
    def __init__(self, inp_dim, contraction_dim, heads, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.attn = AttentionBlock(
            inp_dim, out_channels=contraction_dim, num_heads=heads)
        self.register_buffer('mask', build_local_attention_mask(
            n=6000, l=64), persistent=False)
        self.pos_bias = RelativeQKBias(l=64, max_positions=6000)
        ff_contract = contraction_dim//2
        self.ff1 = nn.Sequential(nn.Conv1d(inp_dim+contraction_dim, ff_contract, kernel_size=1),
                                 nn.GroupNorm(8, ff_contract),
                                 cGLU(ff_contract))
        self.ff2 = nn.Sequential(nn.Conv1d(inp_dim+contraction_dim*3//2, ff_contract, kernel_size=3, padding=1),
                                 nn.GroupNorm(8, ff_contract),
                                 cGLU(ff_contract))

    def forward(self, x):
        ah = self.dropout(self.attn(x, mask=self.mask,
                          qk_bias=self.pos_bias(x.shape[-1])))
        h = torch.cat([ah, x], dim=1)
        hf = self.dropout(checkpoint(self.ff1, h))
        h = torch.cat([h, hf], dim=1)
        hf = self.dropout(checkpoint(self.ff2, h))
        return torch.cat([h, hf], dim=1)


class ConcatAttentionBlock(TimestepBlock):
    def __init__(self, trunk_dim, contraction_dim, blk_dim, heads, dropout):
        super().__init__()
        self.contraction_dim = contraction_dim
        self.prenorm = nn.GroupNorm(8, trunk_dim)
        self.block1 = SubBlock(
            trunk_dim+blk_dim, contraction_dim, heads, dropout)
        self.block2 = SubBlock(
            trunk_dim+blk_dim+contraction_dim*2, contraction_dim, heads, dropout)
        self.out = nn.Conv1d(contraction_dim*4, trunk_dim,
                             kernel_size=1, bias=False)
        self.out.weight.data.zero_()

    def forward(self, x, blk_emb):
        h = self.prenorm(x)
        h = torch.cat(
            [h, blk_emb.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        h = self.block1(h)
        h = self.block2(h)
        h = self.out(h[:, -self.contraction_dim*4:])
        return h + x


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 hidden_dim,
                 out_dim,
                 num_resolutions,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, hidden_dim, kernel_size=5, stride=2)
        # nn.Embedding
        self.resolution_embedding = ml.Embedding(num_resolutions, hidden_dim)
        # Reduces the relative influence of this embedding from the start.
        self.resolution_embedding.weight.data.mul(.1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(hidden_dim, num_attn_heads,
                        do_checkpoint=do_checkpointing))
            attn.append(ResBlock(hidden_dim, dims=1,
                        checkpointing_enabled=do_checkpointing))
        self.attn = nn.Sequential(*attn)
        self.out = ml.Linear(hidden_dim, out_dim, bias=False)
        self.dim = hidden_dim
        self.do_checkpointing = do_checkpointing

    def forward(self, x, resolution):
        h = self.init(x) + self.resolution_embedding(resolution).unsqueeze(-1)
        h = self.attn(h)
        return self.out(h[:, :, 0])


class TransformerDiffusion(nn.Module):
    """
    A diffusion model composed entirely of stacks of transformer layers. Why would you do it any other way?
    """

    def __init__(
            self,
            resolution_steps=8,
            max_window=384,
            model_channels=1024,
            contraction_dim=256,
            num_layers=8,
            in_channels=256,
            input_vec_dim=1024,
            out_channels=512,  # mean and variance
            time_embed_dim=256,
            time_proj_dim=64,
            cond_proj_dim=256,
            num_heads=4,
            dropout=0,
            use_fp16=False,
            # Parameters for regularization.
            # This implements a mechanism similar to what is used in classifier-free training.
            unconditioned_percentage=.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.time_embed_dim = time_embed_dim
        self.out_channels = out_channels
        self.dropout = dropout
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.resolution_steps = resolution_steps
        self.max_window = max_window
        self.preprocessed = None

        self.time_embed = nn.Sequential(
            linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_proj_dim),
        )
        self.prior_time_embed = nn.Sequential(
            linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_proj_dim),
        )
        # nn.Embedding
        self.resolution_embed = ml.Embedding(resolution_steps, time_proj_dim)
        self.conditioning_encoder = ConditioningEncoder(
            in_channels, model_channels, cond_proj_dim, resolution_steps, num_attn_heads=model_channels//64)
        self.unconditioned_embedding = nn.Parameter(
            torch.randn(1, cond_proj_dim))

        self.inp_block = conv_nd(
            1, in_channels+input_vec_dim, model_channels, 3, 1, 1)
        self.layers = TimestepEmbedSequential(*[ConcatAttentionBlock(model_channels, contraction_dim, time_proj_dim*3 + cond_proj_dim,
                                                                     num_heads, dropout) for _ in range(num_layers)])

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(1, model_channels,
                        out_channels, 3, padding=1)),
        )

        self.debug_codes = {}

    def input_to_random_resolution_and_window(self, x, ts, diffuser):
        """
        This function MUST be applied to the target *before* noising. It returns the reduced, re-scoped target as well
        as caches an internal prior for the rescoped target which will be used in training.
        Args:
            x: Diffusion target
        """
        resolution = randrange(0, self.resolution_steps)
        resolution_scale = 2 ** resolution
        s = F.interpolate(x, scale_factor=1/resolution_scale, mode='nearest')
        s_diff = s.shape[-1] - self.max_window
        if s_diff > 1:
            start = randrange(0, s_diff)
            s = s[:, :, start:start+self.max_window]
        s_prior = F.interpolate(s, scale_factor=.25, mode='nearest')
        s_prior = F.interpolate(s_prior, size=(
            s.shape[-1],), mode='linear', align_corners=True)

        # Now diffuse the prior randomly between the x timestep and 0.
        adv = torch.rand_like(ts.float())
        t_prior = (adv * ts).long() - 1
        # The t_prior-1 below is an important detail: it forces s_prior to be unmodified for ts=0. It also means that t_prior is not on the same timescale as ts (instead it is shifted by 1).
        s_prior_diffused = diffuser.q_sample(
            s_prior, t_prior-1, torch.randn_like(s_prior), allow_negatives=True)

        self.preprocessed = (s_prior_diffused, t_prior, torch.tensor(
            [resolution] * x.shape[0], dtype=torch.long, device=x.device))
        return s

    def forward(self, x, timesteps, prior_timesteps=None, x_prior=None, resolution=None, conditioning_input=None, conditioning_free=False):
        """
        Predicts the previous diffusion timestep of x, given a partially diffused low-resolution prior and a conditioning
        input.

        All parameters are optional because during training, input_to_random_resolution_and_window is used by a training
        harness to preformat the inputs and fill in the parameters as state variables.

        Args:
            x: Prediction prior.
            timesteps: Number of timesteps x has been diffused for.
            prior_timesteps: Number of timesteps x_prior has been diffused for. Must be <= timesteps for each batch element. If nothing is specified, then [0] is assumed, e.g. a fully diffused prior.
            x_prior: A low-resolution prior that guides the model.
            resolution: Integer indicating the operating resolution level. '0' is the highest resolution.
            conditioning_input: A semi-related (un-aligned) conditioning input which is used to guide diffusion. Similar to a class input, but hooked to a learned conditioning encoder.
            conditioning_free: Whether or not to ignore the conditioning input.
        """
        conditioning_input = x_prior if conditioning_input is None else conditioning_input

        if resolution is None:
            # This is assumed to be training.
            assert self.preprocessed is not None, 'Preprocessing function not called.'
            assert x_prior is None, 'Provided prior will not be used, instead preprocessing output will be used.'
            x_prior, prior_timesteps, resolution = self.preprocessed
            self.preprocessed = None
        else:
            assert x.shape[-1] > x_prior.shape[-1] * \
                3.9, f'{x.shape} {x_prior.shape}'
            if prior_timesteps is None:
                # This is taken to mean a fully diffused prior was given.
                # Assuming batch_size=1 for inference.
                prior_timesteps = torch.tensor([0], device=x.device)
            x_prior = F.interpolate(x_prior, size=(
                x.shape[-1],), mode='linear', align_corners=True)
        assert torch.all(timesteps - prior_timesteps >=
                         0), f'Prior timesteps should always be lower (more resolved) than input timesteps. {timesteps}, {prior_timesteps}'

        if conditioning_free:
            code_emb = self.unconditioned_embedding.repeat(x.shape[0], 1)
        else:
            MIN_COND_LEN = 200
            MAX_COND_LEN = 1200
            if self.training and conditioning_input.shape[-1] > MAX_COND_LEN:
                clen = randrange(MIN_COND_LEN, MAX_COND_LEN)
                gap = conditioning_input.shape[-1] - clen
                cstart = randrange(0, gap)
                conditioning_input = conditioning_input[:,
                                                        :, cstart:cstart+clen]
            code_emb = self.conditioning_encoder(
                conditioning_input, resolution)

        # Mask out the conditioning input and x_prior inputs for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand(
                (x.shape[0], 1), device=x.device) < self.unconditioned_percentage
            code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(
                code_emb.shape[0], 1), code_emb)

        with torch.autocast(x.device.type, enabled=self.enable_fp16):
            time_emb = self.time_embed(
                timestep_embedding(timesteps, self.time_embed_dim))
            prior_time_emb = self.prior_time_embed(
                timestep_embedding(prior_timesteps, self.time_embed_dim))
            res_emb = self.resolution_embed(resolution)
            blk_emb = torch.cat(
                [time_emb, prior_time_emb, res_emb, code_emb], dim=1)

            h = torch.cat([x, x_prior], dim=1)
            h = self.inp_block(h)
            for layer in self.layers:
                h = checkpoint(layer, h, blk_emb)

        h = h.float()
        out = self.out(h)

        # Defensively involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        unused_params = [self.unconditioned_embedding]
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        out = out + extraneous_addition * 0

        return out

    def get_grad_norm_parameter_groups(self):
        attn1 = list(itertools.chain.from_iterable(
            [lyr.block1.attn.parameters() for lyr in self.layers]))
        attn2 = list(itertools.chain.from_iterable(
            [lyr.block2.attn.parameters() for lyr in self.layers]))
        ff1 = list(itertools.chain.from_iterable([lyr.block1.ff1.parameters() for lyr in self.layers] +
                                                 [lyr.block1.ff2.parameters() for lyr in self.layers]))
        ff2 = list(itertools.chain.from_iterable([lyr.block2.ff1.parameters() for lyr in self.layers] +
                                                 [lyr.block2.ff2.parameters() for lyr in self.layers]))
        blkout_layers = list(itertools.chain.from_iterable(
            [lyr.out.parameters() for lyr in self.layers]))
        groups = {
            'prenorms': list(itertools.chain.from_iterable([lyr.prenorm.parameters() for lyr in self.layers])),
            'blk1_attention_layers': attn1,
            'blk2_attention_layers': attn2,
            'attention_layers': attn1 + attn2,
            'blk1_ff_layers': ff1,
            'blk2_ff_layers': ff2,
            'ff_layers': ff1 + ff2,
            'block_out_layers': blkout_layers,
            'out': list(self.out.parameters()),
            'x_proj': list(self.inp_block.parameters()),
            'layers': list(self.layers.parameters()),
            'time_embed': list(self.time_embed.parameters()),
            'prior_time_embed': list(self.prior_time_embed.parameters()),
            'resolution_embed': list(self.resolution_embed.parameters()),
        }
        return groups

    def before_step(self, step):
        scaled_grad_parameters = list(itertools.chain.from_iterable(
            [lyr.out.parameters() for lyr in self.layers]))
        # Scale back the gradients of the blkout and prenorm layers by a constant factor. These get two orders of magnitudes
        # higher gradients. Ideally we would use parameter groups, but ZeroRedundancyOptimizer makes this trickier than
        # directly fiddling with the gradients.
        for p in scaled_grad_parameters:
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad *= .2


@register_model
def register_transformer_diffusion13(opt_net, opt):
    return TransformerDiffusion(**opt_net['kwargs'])


def test_tfd():
    from models.diffusion.gaussian_diffusion import get_named_beta_schedule
    from models.diffusion.respace import SpacedDiffusion, space_timesteps
    diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [4000]), model_mean_type='epsilon',
                               model_var_type='learned_range', loss_type='mse',
                               betas=get_named_beta_schedule('linear', 4000))
    clip = torch.randn(2, 256, 10336)
    cond = torch.randn(2, 256, 10336)
    ts = torch.LongTensor([0, 0])
    model = TransformerDiffusion(in_channels=256, model_channels=1024, contraction_dim=512,
                                 num_heads=512//64, input_vec_dim=256, num_layers=12, dropout=.1,
                                 unconditioned_percentage=.6)
    model.get_grad_norm_parameter_groups()
    for k in range(100):
        x = model.input_to_random_resolution_and_window(clip, ts, diffuser)
        model(x, ts, conditioning_input=cond)


def remove_conditioning(sd_path):
    sd = torch.load(sd_path)
    del sd['unconditioned_embedding']
    torch.save(sd, sd_path.replace('.pth', '') + '_fixed.pth')


if __name__ == '__main__':
    # remove_conditioning('X:\\dlas\\experiments\\train_music_diffusion_multilevel_sr_pre\\models\\12500_generator.pth')
    test_tfd()
