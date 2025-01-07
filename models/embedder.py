import numpy as np
import torch


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x, *args, **kwargs: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        def _weight_fn(alpha, freq_idx: int, N_freqs: int = N_freqs):
            # https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/803291bd0ee91c7c13fb5cc42195383c5ade7d15/model/barf.py#L259C15-L259C16
            assert alpha >= 0 and alpha <= 1, alpha
            assert freq_idx >= 0 and freq_idx < N_freqs, freq_idx
            weight = (1 - (torch.clamp(alpha * N_freqs - freq_idx, 0, 1) * np.pi).cos()) / 2.
            return weight

        for i_freq, freq in enumerate(freq_bands):
            for p_fn in self.kwargs['periodic_fns']:
                def _embd_fn(x, alpha, w_fn=_weight_fn, p_fn=p_fn, i_freq=i_freq, freq=freq):
                    w = w_fn(alpha=alpha, freq_idx=i_freq)
                    e = p_fn(x * freq)
                    return w * e

                embed_fns.append(_embd_fn)
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs, alpha: float = 1.):
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32, device=inputs.device)
        return torch.cat([fn(x=inputs, alpha=alpha) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, alpha=1., eo=embedder_obj):
        return eo.embed(inputs=x, alpha=alpha)

    return embed, embedder_obj.out_dim
