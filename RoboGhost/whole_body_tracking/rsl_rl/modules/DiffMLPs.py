import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
from ..diffusions.diffusion import create_diffusion
from ..diffusions.transport import create_transport, Sampler
from ..diffusions.diffusion import create_flow
import torch.nn.functional as F
#################################################################################
#                                     DiffMLPs                                  #
#################################################################################
class DiffMLPs_DDPM(nn.Module):
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, learn_sigma=False):
        super(DiffMLPs_DDPM, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2 if learn_sigma else target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")
    def forward(self, target, z1, mask=None):
    # def forward(self, target, z1, z2, z3, z4, mask=None):
        
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        # mask = torch.bernoulli(torch.ones(z1.size(0), device=z1.device) * 0.1).view(z1.size(0), 1)
        # z1 = z1 * (1. - mask)

        model_kwargs = dict(c1=z1)
        # model_kwargs = dict(c1=z1, c2=z2, c3=z3, c4=z4)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        # print(loss)
        return loss.mean()

    def sample(self, z1, temperature=1.0, cfg=1.0):
        device = z1.device
        noise = torch.randn(z1.shape[0], self.in_channels).to(device)
        model_kwargs = dict(c1=z1, cfg_scale=cfg)
        sample_fn = self.net.forward_with_cfg
        sampled_token_latent = self.gen_diffusion.ddim_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            eta=0.0
        )
        return sampled_token_latent

class DiffMLPs_DDPM_CFG(nn.Module):
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, learn_sigma=False):
        super(DiffMLPs_DDPM_CFG, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2 if learn_sigma else target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")
    def forward(self, target, z1, mask=None):

        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        mask = torch.bernoulli(torch.ones(z1.size(0), device=z1.device) * 0.1).view(z1.size(0), 1)
        z1 = z1 * (1. - mask)
        model_kwargs = dict(c1=z1)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z1, temperature=1.0, cfg=1.0):
        device = z1.device
        z1 = torch.cat([z1, torch.zeros_like(z1)], dim=0)
        if not cfg == 1.0:
            noise = torch.randn(z1.shape[0] // 2, self.in_channels).to(device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c1=z1, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg_x0
        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )
        sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
        return sampled_token_latent

class DiffMLPs_FM(nn.Module):
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, learn_sigma=False):
        super(DiffMLPs_FM, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2 if learn_sigma else target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
        )
        self.train_diffusion = create_flow()
        self.gen_diffusion = create_flow()
    def forward(self, target, z, mask=None):
        model_kwargs = dict(c1=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, None, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()
    def sample(self, z1, temperature=1.0, cfg=1.0):
        device = z1.device
        noise = torch.randn(z1.shape[0], self.in_channels).to(device)
        # model_kwargs = dict(c1=z1, c2=z2, cfg_scale=cfg)
        model_kwargs = dict(c1=z1, cfg_scale=cfg)
        sample_fn = self.net.forward_with_cfg
            
        ode_kwargs = {}
        ode_kwargs["method"] = "dopri5"
        ode_kwargs["return_x_est"] = False
        ode_kwargs["return_x_est_num"] = None
        ode_kwargs["step_size"] = 0.01
        ode_kwargs["atol"] = 1e-5
        ode_kwargs["rtol"] = 1e-5
        ode_kwargs["edit_till"] = 1.0
        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            skip_timesteps=0, init_image=None, dump_steps=None, const_noise=False, 
            sample_steps=400, ode_kwargs=ode_kwargs,
        )

        return sampled_token_latent

class DiffMLPs_SiT(nn.Module):
    def __init__(self, target_channels, z_channels, depth, width):
        super(DiffMLPs_SiT, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
        )

        self.train_diffusion = create_transport() # default to linear, velocity prediction
        self.gen_diffusion = Sampler(self.train_diffusion)

    def forward(self, target, z, mask=None):
        model_kwargs = dict(c1=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):
        device = z.device
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).to(device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c1=z, cfg_scale=cfg)
            model_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).to(device)
            model_kwargs = dict(c1=z)
            model_fn = self.net.forward
        sample_fn = self.gen_diffusion.sample_ode()  # default to ode sampling
        sampled_token_latent = sample_fn(noise, model_fn, **model_kwargs)[-1]
        return sampled_token_latent

#################################################################################
#                                  DiffMLPs Zoos                                #
#################################################################################
def diffmlps_ddpm_xl(**kwargs):
    return DiffMLPs_DDPM(depth=4, width=256, num_sampling_steps="2", learn_sigma=False, **kwargs)
def diffmlps_fm_xl(**kwargs):
    return DiffMLPs_FM(depth=8, width=256, num_sampling_steps="2", learn_sigma=False, **kwargs)
def diffmlps_sit_xl(**kwargs):
    return DiffMLPs_SiT(depth=16, width=256, **kwargs)
def diffmlps_ddpm_cfg(**kwargs):
    return DiffMLPs_DDPM_CFG(depth=4, width=256, num_sampling_steps="50", learn_sigma=False, **kwargs)
DiffMLPs_models = {
    'DDPM-XL': diffmlps_ddpm_xl, 'SiT-XL': diffmlps_sit_xl, 'DDPM-CFG': diffmlps_ddpm_cfg, 'FM': diffmlps_fm_xl
}

#################################################################################
#                                Inner Architectures                            #
#################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        try:
            args = t[:, None].float() * freqs[None]
        except:
            args = t * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# class TimestepEmbedder(nn.Module):
#     """
#     Embeds scalar timesteps into vector representations.
#     """
#     def __init__(self, hidden_size, frequency_embedding_size=256):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(frequency_embedding_size, hidden_size, bias=True),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size, bias=True),
#         )
#         self.frequency_embedding_size = frequency_embedding_size

#     @staticmethod
#     def timestep_embedding(t, dim, max_period=10000):
#         """
#         Create sinusoidal timestep embeddings.
#         :param t: a 1-D Tensor of N indices, one per batch element.
#                           These may be fractional.
#         :param dim: the dimension of the output.
#         :param max_period: controls the minimum frequency of the embeddings.
#         :return: an (N, D) Tensor of positional embeddings.
#         """
#         # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
#         half = dim // 2
#         freqs = torch.exp(
#             -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#         ).to(device=t.device)
#         try:
#             args = t[:, None].float() * freqs[None]
#         except:
#             args = t * freqs[None]
#         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#         if dim % 2:
#             embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#         return embedding

#     def forward(self, t):
#         t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
#         t_emb = self.mlp(t_freq)
#         return t_emb

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output p_sample_loopTensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed1 = nn.Linear(1414-270, model_channels)


        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c1):
    # def forward(self, x, t, c1):
    # def forward(self, x, t, c1, c2, c3, c4):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        try:
            c1 = self.cond_embed1(c1)
        except:
            print(c1.shape)
        # c2 = self.cond_embed2(c2)
        # c3 = self.cond_embed3(c3)
        # c4 = self.cond_embed4(c4)
        y = t + c1
        # y = t + c1 + c2 + c3 + c4
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c1, cfg_scale):
    # def forward_with_cfg(self, x, t, c1, c2, cfg_scale):
        model_out = self.forward(x, t, c1)
        return model_out

    def forward_with_cfg_x0(self, x, t, c1, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c1)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
