
import torch.nn as nn
import torch.nn.functional as F

from network.pointnet2_cpp_utils import PointNetSetAbstractionCPP, PointNetSetAbstractionCPPGATT
import torch

from tools.options import Options
opt = Options().parse()

class PointNet2ClsSsgCPP(nn.Module):

    # basic pointnet++ backbone


    def __init__(self,num_class,normal_channel=True):
        super(PointNet2ClsSsgCPP, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        # we use 2 set abstraction graph attention layers
        self.sa1 = PointNetSetAbstractionCPPGATT(
            npoint=384, radius=0.2, nsample=24, 
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False,
            mode='max')
        self.sa2 = PointNetSetAbstractionCPPGATT(
            npoint=64, radius=0.4, nsample=48, 
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False,
            mode='max')


        # this layer only contains MLP
        self.sa3 = PointNetSetAbstractionCPP(
            npoint=None, radius=None, nsample=None, 
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True,
            mode='None')

        # another MLP
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)



        self.fcs_list = []
        for i in range(opt.grid_size**2):
            fc = nn.Linear(256, num_class)
            self.fcs_list.append(fc)
        self.fcs_list = nn.ModuleList(self.fcs_list)



    def forward(self, lidar_float32):
        if lidar_float32.shape[-1]==3:
            lidar_float32 = lidar_float32.permute(0,2,1) # [32,3,4096]

        B, _, _ = lidar_float32.shape
        if self.normal_channel:
            norm = lidar_float32[:, 3:, :]
            lidar_float32 = lidar_float32[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(lidar_float32, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # [32,256,128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # [32,1024,1]

        x = l3_points

        # x:[b,c,64]
        x = x.permute(0,2,1)
        x = self.fc1(x)
        x = x.permute(0,2,1)
        x = self.bn1(x)
        x = F.relu(x)


        # x:[b,c,64]
        x = x.permute(0,2,1)
        x = self.fc2(x)
        x = x.permute(0,2,1)
        x = self.bn2(x)
        x = F.relu(x)


        x = x.permute(0,2,1)

        out = []
        for each_fc in self.fcs_list:
            out.append(each_fc(x))
        out = torch.stack(out, dim=1) # [b,25,6]
        out = out.squeeze(1)

        
        l2_xyz = l2_xyz.permute(0,2,1) # [32,64,3]

        return out, l2_xyz



