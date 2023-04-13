
import network.pointnet2_torch_cuda11.pointnet2_utils as pointnet2_utils
import network.pointnet2_torch_cuda11.pytorch_utils as pt_utils

# from . import pointnet2_utils as pointnet2_utils
# from . import pytorch_utils as pt_utils

import torch
import torch.nn as nn
import torch.nn.functional as F



class FPSBallQueryModule(nn.Module):
    def __init__(self, num_input_points, num_centroids, num_neighbors, radius,
        sample_uniformly
    ) -> None:
        super(FPSBallQueryModule, self).__init__()

        self.grouper = pointnet2_utils.QueryAndGroup(
            radius=radius, 
            nsample=num_neighbors, 
            use_xyz=True, 
            sample_uniformly=sample_uniformly
        )

        self.num_input_points = num_input_points
        self.num_centroids = num_centroids
        self.num_neighbors = num_neighbors
        self.radius = radius
        
    def forward(self, xyz):
        # xyz:[b,1000,3]

        xyz_flipped = xyz.transpose(1, 2).contiguous() # [2,3,1000]
        xyz_centroids = pointnet2_utils.gather_operation(
            xyz_flipped,
            pointnet2_utils.furthest_point_sample(xyz, self.num_centroids)
        ) # [2,3,500]
        xyz_centroids = xyz_centroids.transpose(1, 2).contiguous() # [2,500,3]

        new_xyz = self.grouper(
            xyz, xyz_centroids
        )  # (b, 3, num_input_points, num_centroids)

        new_xyz = new_xyz.permute(0,2,3,1) # (b, num_input_points, num_centroids, 3 )

        return new_xyz, xyz_centroids




if __name__ == "__main__":
    from torch.autograd import Variable
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    num_points = 1000

    xyz = torch.randn(2, num_points, 3).cuda()
    xyz_features = torch.randn(2, num_points, 6).cuda()

    num_sampled_points = 500

    # test_module = PointnetSAModuleMSG(
    #     npoint=500, 
    #     radii=[5.0, 10.0], 
    #     nsamples=[12, 36], 
    #     mlps=[[num_points, 11], 
    #         [num_points, 13]]
    # )
    
    
    xyz_flipped = xyz.transpose(1, 2).contiguous() # [2,3,1000]
    new_xyz = pointnet2_utils.gather_operation(
        xyz_flipped,
        pointnet2_utils.furthest_point_sample(xyz, num_sampled_points)
    ) # [2,3,500]
    new_xyz = new_xyz.transpose(1, 2).contiguous() 

    grouper = pointnet2_utils.QueryAndGroup(
        radius=0.1, nsample=18, use_xyz=True, sample_uniformly=False)

    # input xyz[b,num_points,3], new_xyz[b,num_sampled_points]
    new_features = grouper(
        xyz, new_xyz
    )  # (B, C, npoint, nsample)


    a=1
    
