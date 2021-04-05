import numpy as np
import torch


def get_dists(points1, points2):
    """
    Calculate dists between two group points
    计算两组点之间的距离 (x - y) = sqrt(x^2 + y^2 - 2*x*y)
    :param point1: 第一组点 (B,M,C) 其中M为点个数
    :param point2: 第二组点 (B,N,C) 其中N为点个数
    :return:
    """
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists)
    return torch.sqrt(dists).float()


def gather_points(points, inds):
    """

    :param points: (B,N,C)
    :param inds: (B,M) or (B,M,K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    """
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])  # inds_shape[1:] = [1] or [1,1] --> inds_shape [B,1] or [B,1,1]
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1  # repeat_shape --> [1,M] or [1,M,K]
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    # batchlists --> [B,M] or [B,M,K]
    return points[batchlists, inds, :]


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
