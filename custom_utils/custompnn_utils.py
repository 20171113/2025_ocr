import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

import numpy as np
from sklearn.metrics import f1_score, accuracy_score




class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone=None, n_way=10, normalize=True, proto=None, adapt=False):
        super(PrototypicalNetworks, self).__init__()
        if backbone:
            self.backbone = backbone
        else:
            resnet_pt = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
            resnet_pt.fc = nn.Flatten()
            self.backbone = resnet_pt
        self.n_way = n_way
        self.normalize = normalize
        self.proto = proto
        self.adapt = adapt
        if self.adapt:
            self.adapter = ResidualAdapter()
        
    def forward(self, images, dist=None):
        if self.adapt:
            self.proto = self.adapter(self.proto)
        z = F.normalize(self.backbone.forward(images))
        if dist == None:
            dists = self.euclidean_distance(z, self.proto)  # [Q, N]
        else:
            dists = dist(x, y)
        return -dists
    

    def euclidean_distance(self, x, y):
        n = x.shape[0]  # Q
        m = y.shape[0]  # N
        d = x.shape[1]
        assert d == y.shape[1]

        # x -> [Q, 1, D], y -> [1, N, D]
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)
    
    def GetProto(self):
        return self.proto
    
    def update_proto(self, images, pred, momentum = 0.9):
        z = F.normalize(self.backbone.forward(images)) # bsz, emb_dim
        count_list = torch.tensor([(pred==label).sum() for label in range(10)]).to(self.proto.device) # list of len 10
        z_proto = torch.cat([
            nn.functional.normalize(z[torch.nonzero(pred == label)].mean(0)) if count_list[label]!=0 else torch.zeros(1,z.shape[-1]).to(self.proto.device) for label in range(self.n_way)
        ]).to(self.proto.device)
        momentum_ = count_list*(momentum/count_list.sum()).to(self.proto.device) # list of len 10
        proto_new = self.proto*(1-momentum_).unsqueeze(1) + z_proto*momentum_.unsqueeze(1)
        self.proto = F.normalize(proto_new)
        return None
    
    
    
    


def partition_data_(inputs, labels, n_way = 10):
    indices = []
    for i in range(n_way):
        idx = torch.where(labels==i)[0][:1000].cpu().numpy()
        indices.append(idx)
    indices = np.array(indices)
    indices = np.concatenate(indices)
    return inputs[indices,:,:,:], labels[indices]




class ResidualAdapter(nn.Module):
    def __init__(self, dim=512, h_dim=64):
        super().__init__()
        self.down = nn.Linear(dim, h_dim)
        self.relu = nn.ReLU()
        self.up = nn.Linear(h_dim, dim)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    def forward(self, x):
        #print(self.up(self.relu(self.down(x))).sum())
        return x + self.up(self.relu(self.down(x)))