import torch.nn as nn

from .base_backbone import BaseBackBone



'''
该代码用于获得VGG主干特征提取网络的输出。
输入变量i代表的是输入图片的通道数，通常为3。

一般来讲，输入图像为(300, 300, 3)，随着base的循环，特征层变化如下：
300,300,3 -> 300,300,64 -> 300,300,64 -> 150,150,64 -> 150,150,128 -> 150,150,128 -> 75,75,128 -> 75,75,256 -> 75,75,256 -> 75,75,256 
-> 38,38,256 -> 38,38,512 -> 38,38,512 -> 38,38,512 -> 19,19,512 ->  19,19,512 ->  19,19,512 -> 19,19,512
到base结束，我们获得了一个19,19,512的特征层

之后进行pool5、conv6、conv7。
'''
class VGG(BaseBackBone):
    base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512] #index 16
    def __init__(self):
        super().__init__()
        layers = self.get_whole_layers()
        self.model = nn.ModuleList(layers)
        self.index_list = [12, 21]

    @classmethod
    def get_vgg_backbone(cls, i = 3):
        layers = []
        in_channels = i
        for v in cls.base:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return layers

    @classmethod
    def get_extras(cls, i=1024):
        layers = []
        in_channels = i

        # Block 6
        # 19,19,1024 -> 10,10,512
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

        # Block 7
        # 10,10,512 -> 5,5,256
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

        # Block 8
        # 5,5,256 -> 3,3,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

        # Block 9
        # 3,3,256 -> 1,1,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]


        return layers

    def get_whole_layers(self):
        vgg_layer_list = VGG.get_vgg_backbone()
        extra_layer_list = VGG.get_extras()
        return vgg_layer_list.extend(extra_layer_list)

    def forward(self, x):
        '''
        x(bsz, 3, h, w)
        '''
        for module in self.model:
            x = module(x)
        return x