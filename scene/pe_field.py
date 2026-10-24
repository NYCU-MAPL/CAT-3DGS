from typing import Optional

import torch
import torch.nn as nn


def con_normalize_aabb(pts, aabb, con_aabb):
    scale = (aabb[1] - aabb[0]) / (con_aabb[1] - con_aabb[0])
    normalized_pts = (pts - con_aabb[0]) * scale + aabb[0]
    return normalized_pts

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def positional_embedding(value, L=10):
    """Fourier features of a scalar coordinate: [sin(2^i*pi*v), cos(2^i*pi*v)] for i < L."""
    embed = []
    for i in range(L):
        embed.append(torch.sin(2**i * torch.pi * value).unsqueeze(-1))
        embed.append(torch.cos(2**i * torch.pi * value).unsqueeze(-1))
    return torch.cat(embed, dim=-1)


class PEField(nn.Module):
    """Hyperprior input: per-axis positional embedding of the anchor position.

    Replaces the learnable tri-plane. It has no latent to transmit, so its rate is 0.
    """
    def __init__(
        self,
        bounds,
        planeconfig,
        multires,
        contract,
        comp_iter,
        L: int = 10,
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.resolutions = planeconfig["resolution"]
        self.init_itr = comp_iter

        self.rotation_matrix = nn.Parameter(torch.eye(3), requires_grad=False)
        self.pca_mean = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.variance = nn.Parameter(torch.ones(3), requires_grad=False)
        self.con_aabb = nn.Parameter(torch.ones([2,3]), requires_grad=False)

        self.L = L
        self.feat_dim = 3 * 2 * L
        print("feature_dim:",self.feat_dim)

        self.contract = contract

    def set_aabb(self,rotated_points,xyz_max, xyz_min):
        xyz_max = xyz_max.flatten().cpu().detach().numpy()
        xyz_min = xyz_min.flatten().cpu().detach().numpy()
        if self.contract == True:
            pts = self.contract_to_unisphere(rotated_points.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
            xyz_min = torch.min(pts, dim=0, keepdim=True)[0].detach()
            xyz_max = torch.max(pts, dim=0, keepdim=True)[0].detach()
            self.con_aabb = nn.Parameter(torch.cat((xyz_min, xyz_max), dim=0),requires_grad=False)
            print("after contract:",xyz_min,xyz_max)
            xyz_max = [1.0, 1.0, 1.0]
            xyz_min = [-1.0, -1.0, -1.0]
        aabb = torch.tensor([
            xyz_min,
            xyz_max
        ], device='cuda')
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("PE field: set aabb=",self.aabb)

    def set_rotation_matrix(self, rotation_matrix, mean, variance):
        self.rotation_matrix = nn.Parameter(rotation_matrix, requires_grad=False)
        self.pca_mean = nn.Parameter(mean, requires_grad=False)
        self.variance = nn.Parameter(variance, requires_grad=False)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None, itr=-1):
        if self.contract == True:
            pts = con_normalize_aabb(pts, self.aabb,self.con_aabb)
        else:
            pts = normalize_aabb(pts, self.aabb)

        pts = pts.reshape(-1, pts.shape[-1])
        pex = positional_embedding(pts[:,0], L=self.L)
        pey = positional_embedding(pts[:,1], L=self.L)
        pez = positional_embedding(pts[:,2], L=self.L)
        features = torch.cat([pex, pey, pez], dim=-1)

        return features, 0

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None, itr=-1):
        if self.contract:
            pts = pts - self.pca_mean
            pts = torch.matmul(pts, self.rotation_matrix.detach())
            pts = pts / self.variance.detach()

            pts = self.contract_to_unisphere(pts.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))

        features = self.get_density(pts, itr=itr)

        return features

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x
