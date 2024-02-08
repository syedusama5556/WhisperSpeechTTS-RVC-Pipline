import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

import dlas.torch_intermediary as ml
from dlas.models.arch_util import default, ema_inplace, l2norm, sample_vectors


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') - \
                rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means

# distance types


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.8,
        eps=1e-5
    ):
        super().__init__()
        self.decay = decay
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed.clone())

    def init_embed_(self, data):
        embed = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        modified_codebook = torch.where(mask[..., None], sample_vectors(
            samples, self.codebook_size), self.embed)
        self.embed.data.copy_(modified_codebook)

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        embed = self.embed.t()

        if not self.initted:
            self.init_embed_(flatten)

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(x.dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(
                self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.8,
        eps=1e-5
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(torch.randn(codebook_size, dim))
        else:
            embed = torch.zeros(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('embed', embed)

    def init_embed_(self, data):
        embed = kmeans(data, self.codebook_size,
                       self.kmeans_iters, use_cosine_sim=True)
        self.embed.data.copy_(embed)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        samples = l2norm(samples)
        modified_codebook = torch.where(mask[..., None], sample_vectors(
            samples, self.codebook_size), self.embed)
        self.embed.data.copy_(modified_codebook)

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        flatten = l2norm(flatten)

        if not self.initted:
            self.init_embed_(flatten)

        embed = l2norm(self.embed)
        dist = flatten @ embed.t()
        embed_ind = dist.max(dim=-1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(0)
            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = flatten.t() @ embed_onehot
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(
                zero_mask[..., None], embed, embed_normalized)
            ema_inplace(self.embed, embed_normalized, self.decay)

        return quantize, embed_ind

# main class


class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        n_embed=None,
        codebook_dim=None,
        decay=0.8,
        eps=1e-5,
        kmeans_init=False,
        kmeans_iters=10,
        use_cosine_sim=False,
        max_codebook_misses_before_expiry=0
    ):
        super().__init__()
        n_embed = default(n_embed, codebook_size)

        codebook_dim = default(codebook_dim, dim)
        requires_projection = codebook_dim != dim
        self.project_in = ml.Linear(
            dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = ml.Linear(
            codebook_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps

        klass = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        self._codebook = klass(
            dim=codebook_dim,
            codebook_size=n_embed,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps
        )

        self.codebook_size = codebook_size
        self.max_codebook_misses_before_expiry = max_codebook_misses_before_expiry

        if max_codebook_misses_before_expiry > 0:
            codebook_misses = torch.zeros(codebook_size)
            self.register_buffer('codebook_misses', codebook_misses)

    @property
    def codebook(self):
        return self._codebook.codebook

    def decode(self, codes):
        unembed = F.embedding(codes, self._codebook.embed)
        return self.project_out(unembed)

    def expire_codes_(self, embed_ind, batch_samples):
        if self.max_codebook_misses_before_expiry == 0:
            return

        embed_ind = rearrange(embed_ind, '... -> (...)')
        misses = torch.bincount(embed_ind, minlength=self.codebook_size) == 0
        self.codebook_misses += misses

        expired_codes = self.codebook_misses >= self.max_codebook_misses_before_expiry
        if not torch.any(expired_codes):
            return

        self.codebook_misses.masked_fill_(expired_codes, 0)
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self._codebook.replace(batch_samples, mask=expired_codes)

    def forward(self, x):
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)
        commit_loss = F.mse_loss(quantize.detach(), x)

        if self.training:
            quantize = x + (quantize - x).detach()
            self.expire_codes_(embed_ind, x)

        quantize = self.project_out(quantize)
        return quantize, embed_ind, commit_loss
