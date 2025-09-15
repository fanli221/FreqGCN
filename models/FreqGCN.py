import math
import torch
import torch.nn as nn

# -------------------- Core layers --------------------

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.att = nn.Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.att.size(0))
        self.att.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        x: [B, num_nodes, in_features]
        returns: [B, num_nodes, out_features]
        """
        support = torch.matmul(x, self.weight)          # [B, N, F_out]
        out = torch.matmul(self.att, support)           # [B, N, F_out]
        return out + self.bias if self.bias is not None else out


class GCNBlock(nn.Module):
    def __init__(self, dim, node_n):
        super().__init__()
        self.emb_fc = nn.Linear(dim, dim)
        self.gcn1 = GraphConvolution(dim, dim, node_n=node_n)
        self.gcn2 = GraphConvolution(dim, dim, node_n=node_n)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x, embed):
        x = x + self.emb_fc(self.act(embed)).unsqueeze(1)

        x_ = self.norm1(x)
        x_ = self.gcn1(x_)
        x_ = self.act(x_)
        x = x + x_

        x_ = self.norm2(x)
        x_ = self.gcn2(x_)
        x_ = self.act(x_)
        x = x + x_

        return x


class DiffGCN(nn.Module):
    def __init__(self, latent_dim=512, joint_dim=48, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [GCNBlock(latent_dim, joint_dim) for _ in range(num_layers)]
        )

    def forward(self, motion_input, embed):
        # motion_input: [B, N, dim]
        x = motion_input
        for layer in self.layers:
            x = layer(x, embed)
        return x


class HumanGCN(nn.Module):
    def __init__(self, joint_dim=48, latent_dim=512, num_layers=12, num_frames=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.joint_dim = joint_dim

        self.input_process = nn.Conv1d(num_frames, latent_dim, 1)
        self.condition_process = nn.Conv1d(num_frames, latent_dim, 1)

        self.merge = nn.Linear(2 * latent_dim, latent_dim)

        self.gcn = DiffGCN(latent_dim, joint_dim=joint_dim, num_layers=num_layers)
        self.output_process = nn.Conv1d(latent_dim, num_frames, 1)

    def forward(self, x, timesteps, mod):
        emb = timestep_embedding(timesteps, self.latent_dim)

        if mod is None:
            mod = torch.zeros_like(x, device=x.device)

        cond_emb = self.condition_process(mod)   # [B, D, J]
        x_proc = self.input_process(x)          # [B, D, J]

        x_cat = torch.cat((cond_emb, x_proc), dim=1).permute(0, 2, 1)  # [B, J, 2D]
        x_cat = self.merge(x_cat)                                      # [B, J, D]

        out = self.gcn(x_cat, emb)           # [B, J, D]
        out = out.permute(0, 2, 1)           # [B, D, J]
        out = self.output_process(out)       # [B, T, J]
        return out

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    sinusoidal timestep embeddings.
    timesteps: [B]
    return: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
