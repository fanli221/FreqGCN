import copy
import torch
from utils import *
import numpy as np
import math
from copy import deepcopy


def sqrt_beta_schedule(timesteps, s=0.0001):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = 1 - torch.sqrt(t + s)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3., end=3., tau=0.7, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion:
    def __init__(self, noise_steps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 motion_size=(35, 66),
                 device="cuda",
                 padding=None,
                 EnableComplete=True,
                 ddim_timesteps=100,
                 scheduler='Linear',
                 model_type='data',
                 mod_enable=True,
                 mod_test=0.5,
                 dct=None,
                 idct=None,
                 n_pre=10):
        self.noise_steps = noise_steps
        self.beta_start = (1000 / noise_steps) * beta_start
        self.beta_end = (1000 / noise_steps) * beta_end
        self.motion_size = motion_size
        self.device = device

        self.scheduler = scheduler  # 'Cosine', 'Sqrt', 'Linear', 'Sigmoid'
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.ddim_timesteps = ddim_timesteps

        self.model_type = model_type
        self.padding = padding  # 'Zero', 'Repeat', 'LastFrame'
        self.EnableComplete = EnableComplete
        self.mod_enable = mod_enable
        self.mod_test = mod_test

        self.ddim_timestep_seq = np.asarray(
            list(range(0, self.noise_steps, self.noise_steps // self.ddim_timesteps))) + 1
        self.ddim_timestep_prev_seq = np.append(np.array([0]), self.ddim_timestep_seq[:-1])

        self.dct = dct
        self.idct = idct
        self.n_pre = n_pre

    def prepare_noise_schedule(self):
        if self.scheduler == 'Linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.scheduler == 'Cosine':
            return cosine_beta_schedule(self.noise_steps)
        elif self.scheduler == 'Sqrt':
            return sqrt_beta_schedule(self.noise_steps)
        elif self.scheduler == 'Sigmoid':
            return sigmoid_beta_schedule(self.noise_steps)
        else:
            raise NotImplementedError(f"unknown scheduler: {self.scheduler}")

    def noise_motion(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def inpaint_complete(self, step, x, prev_t, traj_dct, mode_dict):
        """
        perform mask completion

        Args:
            step: current diffusion timestep
            x: x in prev_t step
            prev_t:  timestep in prev_t
            traj_dct: DCT coefficient of the traj,
                    shape as [sample_num, n_pre, 3 * joints_num]
            mode_dict: mode helper dict as sample_ddim()

        Returns:
            completed sample
        """
        x_prev_t_known, _ = self.noise_motion(traj_dct, prev_t)  # add noise in DCT domain

        x_prev_t_known = torch.matmul(self.idct[:, :self.n_pre],
                                      x_prev_t_known[:, :self.n_pre])  # return time domain for mask
        x_prev_t_unknown = torch.matmul(self.idct[:, :self.n_pre],
                                        x[:, :self.n_pre])
        x = torch.mul(mode_dict['mask'], x_prev_t_known) + torch.mul((1 - mode_dict['mask']), x_prev_t_unknown)  # mask
        x = torch.matmul(self.dct[:self.n_pre], x)

        return x

    def sample_ddim_progressive(self, model, traj_dct, traj_dct_mod, mode_dict, noise=None):
        """
        Generate samples from the model and yield samples from each timestep.

        Args are the same as sample_ddim()
        Returns a generator contains x_{prev_t}, shape as [sample_num, n_pre, 3 * joints_num]
        """
        model.eval()
        sample_num = mode_dict['sample_num']

        if noise is not None:
            x = noise
        else:
            x = torch.randn((sample_num, self.motion_size[0], self.motion_size[1])).to(self.device)

        with torch.no_grad():
            for i in reversed(range(0, self.ddim_timesteps)):
                t = (torch.ones(sample_num) * self.ddim_timestep_seq[i]).long().to(self.device)
                prev_t = (torch.ones(sample_num) * self.ddim_timestep_prev_seq[i]).long().to(self.device)

                alpha_hat = self.alpha_hat[t][:, None, None]
                alpha_hat_prev = self.alpha_hat[prev_t][:, None, None]

                predicted_noise = model(x, t, mod=traj_dct_mod)

                predicted_x0 = (x - torch.sqrt((1. - alpha_hat)) * predicted_noise) / torch.sqrt(alpha_hat)
                pred_dir_xt = torch.sqrt(1 - alpha_hat_prev) * predicted_noise
                x_prev = torch.sqrt(alpha_hat_prev) * predicted_x0 + pred_dir_xt

                x = x_prev

                if self.EnableComplete is True:
                    x = self.inpaint_complete(i,
                                              x,
                                              prev_t,
                                              traj_dct,
                                              mode_dict)

                yield x
                
    # def sample_ddim_progressive(self, model, traj_dct, traj_dct_mod, mode_dict, noise=None,
    #                         eta=0.6, eta_schedule='const'):
    #     """
    #     仅加入 DDIM 随机性:
    #     - eta=0.0  => 完全等价你现在的确定性 DDIM
    #     - eta>0.0  => 引入一步内随机项，提升多样性
    #     eta_schedule: 'const' 或 'cos'（余弦退火：早期大、后期小）
    #     其余逻辑 (inpaint_complete / 接口 / yield) 均保持不变
    #     """
    #     model.eval()
    #     sample_num = mode_dict['sample_num']

    #     # 初始化 x_T
    #     if noise is not None:
    #         x = noise.to(self.device)
    #     else:
    #         x = torch.randn((sample_num, self.motion_size[0], self.motion_size[1]), device=self.device)

    #     S = self.ddim_timesteps  # 采样步数

    #     with torch.no_grad():
    #         for i in reversed(range(0, S)):
    #             # 当前与前一（更靠近 0）的时间步索引
    #             t     = (torch.ones(sample_num, device=self.device) * self.ddim_timestep_seq[i]).long()
    #             prev_t= (torch.ones(sample_num, device=self.device) * self.ddim_timestep_prev_seq[i]).long()

    #             a_t  = self.alpha_hat[t]      # [B]
    #             a_s  = self.alpha_hat[prev_t] # [B]

    #             # 预测噪声 ε_theta(x_t, t | mod) —— 不改你的调用
    #             predicted_noise = model(x, t, mod=traj_dct_mod)

    #             # 计算 x0
    #             sqrt_at  = torch.sqrt(a_t)[:, None, None]
    #             sqrt_1at = torch.sqrt(1. - a_t)[:, None, None]
    #             predicted_x0 = (x - sqrt_1at * predicted_noise) / (sqrt_at + 1e-8)

    #             # 计算本步 eta_t（常数或余弦退火）
    #             if eta_schedule == 'cos' and S > 1:
    #                 # progress: 0 -> 1 （从早到晚），由于 for 是 reversed，需要翻转一下索引
    #                 progress = (self.ddim_timesteps - 1 - i) / (self.ddim_timesteps - 1)
    #                 eta_t = eta * 0.5 * (1.0 + math.cos(math.pi * progress))  # 早期大、后期趋近 0
    #             else:
    #                 eta_t = eta

    #             # sigma_t（DDIM 论文公式）
    #             # sigma_t = eta_t * sqrt( (1 - a_s)/(1 - a_t) * (1 - a_t/a_s) )
    #             sigma_t = eta_t * torch.sqrt(
    #                 (1. - a_s) / (1. - a_t + 1e-8) * (1. - a_t / (a_s + 1e-8) + 1e-8)
    #             )[:, None, None]

    #             # 系数 c = sqrt(1 - a_s - sigma_t^2)
    #             c = torch.sqrt(torch.clamp(1. - a_s - sigma_t.squeeze(-1).squeeze(-1) ** 2, min=0.0))[:, None, None]

    #             # x_{t-1} = sqrt(a_s) * x0 + c * eps + sigma_t * z
    #             pred_dir_xt = c * predicted_noise
    #             x_prev = torch.sqrt(a_s)[:, None, None] * predicted_x0 + pred_dir_xt

    #             if eta_t > 0.0:
    #                 z = torch.randn_like(x)
    #                 x_prev = x_prev + sigma_t * z

    #             x = x_prev

    #             # 掩码补全（保持你的原逻辑）
    #             if self.EnableComplete is True:
    #                 x = self.inpaint_complete(i, x, prev_t, traj_dct, mode_dict)

    #             yield x

    def sample_ddim(self,
                    model,
                    traj_dct,
                    traj_dct_mod,
                    mode_dict):
        """
        Generate samples from the model.

        Args:
            model: the model to predict noise
            traj_dct: DCT coefficient of the traj,
                shape as [sample_num, n_pre, 3 * joints_num]
            traj_dct_mod: equal to traj_dct or None when no modulation
            mode_dict: a dict containing the following keys:
                 - 'mask': [[1, 1, ..., 0, 0, 0]
                            [1, 1, ..., 0, 0, 0]
                            ...
                            [1, 1, ..., 0, 0, 0]], mask for observation
                 - 'sample_num': sample_num for different modes
                 - 'mode': mode name, e.g. 'pred', 'control'....
                 when mode is 'switch', there are two additional keys:
                 - 'traj_switch': traj to switch to
                 - 'mask_end': [[0, 0, ...., 1, 1, 1]
                                [0, 0, ...., 1, 1, 1]
                                ...
                                [0, 0, ...., 1, 1, 1]], mask for switch
                 when mode is 'control', there are one additional key:
                 - 'traj_fix': retained the fixed part of the traj
                    and current mask will be:
                                [[0, 0, ...., 1, 1, 1]
                                [1, 1, ...., 1, 1, 1]
                                ...
                                [0, 0, ...., 1, 1, 1]]
        Returns: sample
        """
        final = None
        for sample in self.sample_ddim_progressive(model,
                                                   traj_dct,
                                                   traj_dct_mod,
                                                   mode_dict):
            final = sample
        return final
