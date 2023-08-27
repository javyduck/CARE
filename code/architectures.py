import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.lenet import LeNet
from archs.mlp import MLP
from archs.awa_resnet import AWADNN
from archs.neural import NEURAL
from datasets import get_normalize_layer
from torch.nn.functional import interpolate
from torch.sparse import mm
from torch.nn import Softmax
import numpy as np
from train_utils import setup_seed
# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", 'neural','lenet', 'MLP']

def get_architecture(arch: str, dataset: str, classes: int = 50) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if dataset == "AWA":
        model = torch.nn.DataParallel(AWADNN(classes = classes)).cuda()
    elif dataset == "word50_letter":
        model = torch.nn.DataParallel(MLP(n_class = 26)).cuda()
    elif dataset == "word50_word":
        model = torch.nn.DataParallel(MLP(n_class = 50)).cuda()
    elif dataset == "stop_sign":
        if arch == "neural":
            model = NEURAL(12, 3).cuda()
        elif arch == "neural_attribute":
            model = NEURAL(2, 3).cuda()
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)

class GCN(torch.nn.Module):
    def __init__(self, total_num, formula_num, dim = 512):
        super(GCN, self).__init__()
        self.total_num = total_num
        self.dim = dim
        self.conv1 = GCNConv(self.dim, self.dim)
        self.conv2 = GCNConv(self.dim, self.dim)
        self.conv3 = GCNConv(self.dim, 1)
        self.w = nn.Parameter(torch.zeros(1, formula_num)
        self.embedding = nn.Parameter(nn.Linear(self.total_num, self.dim).weight.T)
                              
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.reshape(-1,self.total_num,1) * self.embedding
        x = x.reshape(-1,self.dim)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index).view(-1,self.total_num )
        return x