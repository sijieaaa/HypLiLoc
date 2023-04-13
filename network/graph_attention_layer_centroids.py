
import torch.nn as nn
import torch
import torch.nn.functional as F


from tools.utils import set_seed
from tools.options import Options
opt = Options().parse()
set_seed(7)


class GraphAttentionLayerCentroids(nn.Module):

    # graph attention layer without learnable matrix

    def __init__(self, 
        in_features, 
        hidden_features,
        n_heads=8,
        num_neighbors=None):
        super(GraphAttentionLayerCentroids, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.n_heads = n_heads


        self.W_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        self.W = nn.Linear(in_features, in_features)

        # default: LayerNorm and ReLU
        if opt.gattnorm=='ln':
            self.norm = nn.LayerNorm(in_features)
        if opt.gattactivation=='gelu':
            self.activation = nn.GELU()
        if opt.gattactivation=='relu':
            self.activation = nn.ReLU(True)




    def forward(self, t, h):


        b, c, num_centroids = h.shape
        h = h.permute(0,2,1) # [b,512,128]





        h = self.W(h)

        h = h.view(b, num_centroids, self.n_heads, -1)
        h = h.permute(0,2,1,3) # [b, n_heads, num_centroids, c]
        attention = h @ h.transpose(-2,-1)
        attention = F.softmax(attention, -1)
        h = attention @ h

        out = h
        out = out.permute(0,2,1,3)
        out = out.contiguous()
        out = out.view(b, num_centroids, -1)



        if opt.gattnorm is not None:
            out = self.norm(out)

        if opt.gattactivation is not None:
            out = self.activation(out)


        out = out.permute(0,2,1) # [b,128,512]

     

        return out