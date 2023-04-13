import torch
import torch.nn as nn
from network.graph_attention_layer_fusion import GraphAttentionLayerFusion
from network.pointnet2_cls_ssg_cpp import PointNet2ClsSsgCPP
from tools.options import Options
from tools.utils import set_seed
from hyptorch.nn import *

import torchvision
set_seed(7)
opt = Options().parse()




def l2normalize(x, dim):
    x_l2norm = torch.norm(x, dim=dim, keepdim=True)
    x = x/x_l2norm

    return x



import configs.hyperparameters as hp
from models.sequential_layers import fc_dropout
class RegressionHeadPointLoc(nn.Module):
    def __init__(self):
        super(RegressionHeadPointLoc, self).__init__()
        # self.fc1_translation = fc_dropout(1024, 512, hp.DROPOUT_RATE)
        self.fc2_translation = fc_dropout(512, 128, hp.DROPOUT_RATE)
        self.fc3_translation = fc_dropout(128, 64, hp.DROPOUT_RATE)
        self.translation = nn.Linear(64, 3)

        # self.fc1_rotation = fc_dropout(1024, 512, hp.DROPOUT_RATE)
        self.fc2_rotation = fc_dropout(512, 128, hp.DROPOUT_RATE)
        self.fc3_rotation = fc_dropout(128, 64, hp.DROPOUT_RATE)
        self.rotation = nn.Linear(64, 3)
    def forward(self, x):

        # out1 = self.fc1_translation(x)
        out1 = self.fc2_translation(x)
        out1 = self.fc3_translation(out1)
        out1 = self.translation(out1)

        # out2 = self.fc1_rotation(x)
        out2 = self.fc2_rotation(x)
        out2 = self.fc3_rotation(out2)
        out2 = self.rotation(out2)
        

        out = torch.cat((out1, out2), 1)


        return out








class GATTMLP(nn.Module):
    def __init__(self, in_features):
        super(GATTMLP, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features,in_features),
            nn.LayerNorm(in_features),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.seq(x)

        return x



class BasicBlock(nn.Module):
    def __init__(self, in_features):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_features,in_features,3,1,1)
        self.bn1 = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features,in_features,3,1,1)
        self.bn2 = nn.BatchNorm2d(in_features)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out





class HypLiLoc(nn.Module):
    def __init__(self):
        super(HypLiLoc, self).__init__()
        if opt.lidar_model=='PointNet2ClsSsgCPP':
            lidar_model = PointNet2ClsSsgCPP(num_class=512, normal_channel=False)
        if opt.model=='ResNet34':
            image_model = torchvision.models.resnet34(pretrained=True)
        if opt.model=='ResNet34':
            prj_model = torchvision.models.resnet34(pretrained=True)




        self.lidar_model = lidar_model
        self.image_model = image_model
        self.prj_model = prj_model




        self.lidar_head = RegressionHeadPointLoc()
        self.prj_head = RegressionHeadPointLoc()
        self.fusion_head = RegressionHeadPointLoc()





        self.gatt1 = GraphAttentionLayerFusion(in_features=opt.feat_dim,hidden_features=opt.feat_dim//8,n_heads=8,num_neighbors=None)
        self.gatt1_poin = GraphAttentionLayerFusion(in_features=opt.feat_dim,hidden_features=opt.feat_dim//8,n_heads=8,num_neighbors=None)
        self.gatt1_w = nn.Parameter(torch.ones([2]).float(),requires_grad=True)
        self.gatt2 = GraphAttentionLayerFusion(in_features=opt.feat_dim,hidden_features=opt.feat_dim//8,n_heads=8,num_neighbors=None)
        self.gatt2_poin = GraphAttentionLayerFusion(in_features=opt.feat_dim,hidden_features=opt.feat_dim//8,n_heads=8,num_neighbors=None)
        self.gatt2_w = nn.Parameter(torch.ones([2]).float(),requires_grad=True)


        self.basicblock1 = BasicBlock(in_features=512)
        self.lidarmlp11 = GATTMLP(in_features=512)
        self.lidargatt1 = GraphAttentionLayerFusion(in_features=opt.feat_dim,hidden_features=opt.feat_dim//8,n_heads=8,num_neighbors=None)
        self.lidarmlp12 = GATTMLP(in_features=512)


        self.topoincare = ToPoincare(c=0.1,ball_dim=512,riemannian=False,clip_r=None) # hyperbolic embedding



    def forward_backbone(self, x, model):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        return x





    def forward(self, lidar, prj):
        

        # 3D point cloud features
        lidar_feat, lidar_xyz = self.lidar_model(lidar) # [b,num_centroids,c]
        lidar_feat_org = lidar_feat



        

        # 2D spherical projection features
        prj_feat = self.forward_backbone(prj, self.prj_model)
        prj_feat_org = prj_feat
        b,c,h,w = prj_feat.shape
        prj_feat = prj_feat.flatten(2)
        prj_feat = prj_feat.permute(0,2,1) # [b,hw,c]

        


        lidar_feat_pool = lidar_feat.mean(1).unsqueeze(1)
        prj_feat_pool = prj_feat.mean(1).unsqueeze(1)


        # L2 normalization
        lidar_feat = l2normalize(lidar_feat, dim=-1)
        prj_feat = l2normalize(prj_feat, dim=-1)
        lidar_feat_pool = l2normalize(lidar_feat_pool, dim=-1)
        prj_feat_pool = l2normalize(prj_feat_pool, dim=-1)



        num_pts_lidar = lidar_feat.shape[1]
        num_pts_prj = prj_feat.shape[1]


        # merging
        fusion_feat = torch.cat([lidar_feat, prj_feat, lidar_feat_pool, prj_feat_pool],dim=1)

        # graph attention in the Euclidean space
        fusion_feat_eucl = self.gatt1(fusion_feat)
        # graph attention in the hyperbolic space
        fusion_feat_poin = self.gatt1_poin(self.topoincare(fusion_feat))
        # adaptive fusion
        fusion_feat = fusion_feat_eucl*self.gatt1_w[0] + fusion_feat_poin*self.gatt1_w[1]


        # modal-specific learning for 3D features
        lidar_feat = fusion_feat[:,:num_pts_lidar]
        lidar_feat = self.lidarmlp11(lidar_feat)
        lidar_feat = self.lidargatt1(lidar_feat)
        lidar_feat = self.lidarmlp12(lidar_feat)
        lidar_feat_pool = lidar_feat.mean(1).unsqueeze(1)


        # modal-specific learning for 2D features
        prj_feat = fusion_feat[:,num_pts_lidar:num_pts_lidar+num_pts_prj]
        prj_feat = prj_feat.permute(0,2,1)
        prj_feat = prj_feat.view(b,c,h,w)
        prj_feat = self.basicblock1(prj_feat)
        prj_feat = prj_feat.flatten(2)
        prj_feat = prj_feat.permute(0,2,1) # [b,hw,c]
        prj_feat_pool = prj_feat.mean(1).unsqueeze(1)

        

        # another Euclidean-hyperbolic block
        fusion_feat = torch.cat([lidar_feat, prj_feat, lidar_feat_pool, prj_feat_pool],dim=1)
        fusion_feat_eucl = self.gatt2(fusion_feat)
        fusion_feat_poin = self.gatt2_poin(self.topoincare(fusion_feat))
        fusion_feat = fusion_feat_eucl*self.gatt2_w[0] + fusion_feat_poin*self.gatt2_w[1]






        # output feature vectors for loss computing
        output_lidar = self.lidar_head(lidar_feat_org.mean(1))
        output_prj = self.prj_head(prj_feat_org.flatten(2).permute(0,2,1).mean(1))
        output_fusion = self.fusion_head(fusion_feat.mean(1))



        output_dict = {
            'output_lidar':output_lidar,
            'output_prj':output_prj,
            'output_fusion':output_fusion
        }
        

        return output_dict



