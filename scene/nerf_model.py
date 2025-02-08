import torch
import torch.nn as nn
import numpy as np

class VDGS(nn.Module):
    def __init__(self):
        super().__init__()

        output_size = 3
        self.embed_pos, self.embed_pos_cnl = get_embedder(3, 1)
        self.embed_view, self.embed_view_cnl = get_embedder(10, 3)

        in_cnl =  self.embed_view_cnl + self.embed_pos_cnl
        self.mlp_1 = nn.Sequential(
            nn.Linear(in_cnl, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.head_mul = nn.Sequential(
            nn.Linear(64, output_size), nn.Sigmoid(),
        )
        self.head_offset_factor = nn.Sequential(
            nn.Linear(64, output_size), nn.Sigmoid(),
        )
        self.head_attn = nn.Sequential(
            nn.Linear(64, output_size), nn.Softplus(),
        )
        self.head_bs = nn.Sequential(
            nn.Linear(64, output_size), nn.Softplus(),
        )
        self.head_background = nn.Sequential(
            nn.Linear(64, output_size), nn.Sigmoid(),
        )
        self.mlp_1.apply(init_linear_weights)
        self.head_mul.apply(init_linear_weights)
        self.head_offset_factor.apply(init_linear_weights)
        self.head_attn.apply(init_linear_weights)
        self.head_bs.apply(init_linear_weights)
        self.head_background.apply(init_linear_weights)

    def forward(self, Viewdir, distance):

        distance = self.embed_pos(distance)
        Viewdir = self.embed_view(Viewdir)
        inp = torch.cat([Viewdir, distance], dim=-1)
        x = self.mlp_1(inp)
        mul = self.head_mul(x)
        mul = mul / mul.max()
        offset = self.head_offset_factor(x)
        attn = self.head_attn(x)
        bs = self.head_bs(x)
        background_light = self.head_background(x)
        backscattering = background_light * offset

        return mul, offset, attn, bs, backscattering

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
