import torch
from utils.common import get_dists


def fps(xyz, M):
    """
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    根据最远点采样算法从点云中采样M个点
    :param xyz: 输入的点云 (B,N,3)
    :param M: 要采样M个点
    :return: inds (B,M)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B,), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)    # [0,1,...,B-1]
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :]  # (B,3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids
