'''Darknet in PyTorch.
As seen here https://github.com/fastai/fastai/blob/master/courses/dl2/cifar10-darknet.ipynb
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
    

def conv_layer(ni, nf, ks=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=ks//2),
        nn.BatchNorm2d(nf, momentum=0.01),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Dropout2d(p = 0.05))

class ResLayer(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.conv1=conv_layer(ni, ni//2, ks=1)
        self.conv2=conv_layer(ni//2, ni, ks=3)
        
    def forward(self, x): 
        return x.add(self.conv2(self.conv1(x)))

dropout = 0.5
class Darknet(nn.Module):
    def make_group_layer(self, ch_in, num_blocks, stride=1):
        return [conv_layer(ch_in, int(np.round(ch_in*2)),stride=stride)
               ] + [(ResLayer(int(np.round(ch_in*2)))) for i in range(num_blocks)]

    def __init__(self, num_blocks, num_classes, nf=32):
        nf_initial = nf
        super().__init__()
        layers = [conv_layer(64, nf, ks=3, stride=1)]
        for i,nb in enumerate(num_blocks):
            layers += self.make_group_layer(nf, nb, stride=2-(i==1))
            nf = int(np.round(nf*2))
        layers += [nn.AdaptiveAvgPool2d((1)), Flatten(), 
                   nn.Dropout(dropout), nn.Linear(int((2**len(num_blocks))*nf_initial), num_classes)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x): return self.layers(x)