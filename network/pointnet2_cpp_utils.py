
from network.graph_attention_layer_centroids import GraphAttentionLayerCentroids
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import network.pointnet2_torch_cuda11.pointnet2_utils as pointnet2_utils
from tools.options import Options
opt = Options().parse()
from tools.utils import set_seed
set_seed(7)



# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

def polar2cart(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


def cart2sphere(input_xyz): # [b,num_points,3]
    longtitude = torch.arctan2(input_xyz[:,:,1], input_xyz[:,:,0])
    xy_r = torch.sqrt(input_xyz[:,:,0]**2 + input_xyz[:,:,1]**2)
    latitude = torch.arctan2(input_xyz[:,:,2], xy_r)
    xyz_r = torch.sqrt(input_xyz[:,:,0]**2 + input_xyz[:,:,1]**2 + input_xyz[:,:,2]**2)
    zero = torch.zeros_like(xyz_r)
    output_longlar = torch.stack([longtitude, latitude, xyz_r, zero], dim=-1)
    return output_longlar


def sphere2cart(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)




def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape # [b,4096,3]
    _, S, _ = new_xyz.shape # [b,512,3]
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1]) # [32,512,4096]
    sqrdists = square_distance(new_xyz, xyz)  # [32,512,4096]
    group_idx[sqrdists > radius ** 2] = N # far away points are all 4096
    group_idx = group_idx.sort(dim=-1) # from small to large distance  # [32,512,4096]
    group_idx = group_idx[0][:, :, :nsample] # [32,512,32]
    group_first = group_idx[:, :, 0] # [32,512]
    group_first = group_first.view(B, S, 1)
    group_first = group_first.repeat([1, 1, nsample]) # [32,512,32]
    mask = group_idx == N # top nsample points, but outside radius
    group_idx[mask] = group_first[mask] # 
    return group_idx


def sample_and_group(num_centroids, radius, num_neighbors, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """

    xyz = xyz.contiguous() # [b,4096,3]
    B, num_input_points  = xyz.shape[:2]
    num_centroids = num_centroids


    # FPS sample
    xyz_flipped = xyz.permute(0,2,1).contiguous() # [b,3,4096]
    new_xyz = pointnet2_utils.gather_operation(
        xyz_flipped,
        pointnet2_utils.furthest_point_sample(xyz, num_centroids)
    ) 
    new_xyz = new_xyz.permute(0,2,1) # [b,512,3]




    # ball query cpp
    idx = pointnet2_utils.ball_query(radius*opt.divide_factor, num_neighbors, 
        xyz[:,:,:].contiguous(), new_xyz[:,:,:].contiguous())
    idx = idx.detach().long()
    assert list(idx.shape)==[xyz.shape[0],num_centroids,num_neighbors]

    grouped_xyz = index_points(xyz, idx) 
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, num_centroids, 1, 3) 
    
    if points is not None:
        grouped_points = index_points(points, idx) 
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) 
    else:
        new_points = grouped_xyz_norm


    return new_xyz, new_points, None, None





def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points




class PointNetSetAbstractionCPP(nn.Module):

    # MLP layer

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, mode=None):
        super(PointNetSetAbstractionCPP, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        


    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1) # [32,4096,3]
        if points is not None:
            points = points.permute(0, 2, 1)

        # below is just a place holder to adapt to the architecture
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, new_xyz_second, new_points_second = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points) # [B, npoint, nsample, C+D]



        # MLP
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs): # 1x1conv
            bn = self.mlp_bns[i]
            new_points = conv(new_points)
            new_points = bn(new_points)
            new_points =  F.relu(new_points)


        new_points = new_points.squeeze(-1)



        return new_xyz, new_points






class PointNetSetAbstractionCPPGATT(nn.Module):

    # set absraction graph attention (SAGA) layer

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, mode):
        super(PointNetSetAbstractionCPPGATT, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mode = mode

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        



        self.gatt_layer_centroids = GraphAttentionLayerCentroids(
            in_features=mlp[-1],
            hidden_features=mlp[-1]//8,
            n_heads=8,
            num_neighbors=npoint
        )



    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)


        # set abstraction
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, _, _ = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points) 

        # MLP
        new_points = new_points.permute(0, 3, 2, 1) 
        for i, conv in enumerate(self.mlp_convs): # 1x1conv
            bn = self.mlp_bns[i]
            new_points = conv(new_points)
            new_points = bn(new_points)
            new_points =  F.relu(new_points)


        # max aggregation
        if self.mode=='max':
            new_points = torch.max(new_points, 2) 
            new_points = new_points[0] # [32,128,512]


        new_xyz = new_xyz.permute(0, 2, 1) # [32,3,512]
        

        # graph attention
        new_points = self.gatt_layer_centroids(t=None,h=new_points)
        new_points_second = self.gatt_layer_centroids(t=None,h=new_points)
        new_points = new_points + new_points_second



        return new_xyz, new_points



