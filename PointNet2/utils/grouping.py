import torch
from utils.common import gather_points, get_dists


def ball_query(xyz, new_xyz, radius, K):
    """
    球查询
    :param xyz: 原点云 (B,N,3)
    :param new_xyz: 操作后的点云 (B,M,3)
    :param radius: 查询半径 int
    :param K: int, 限制采样数量
    :return: (B,M,K)
    """
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = get_dists(new_xyz, xyz)
    grouped_inds[dists > radius] = N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    return grouped_inds
