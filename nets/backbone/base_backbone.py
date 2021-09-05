import torch.nn as nn

class BaseBackBone(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError