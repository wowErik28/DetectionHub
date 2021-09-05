import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F

from nets.backbone.vgg import VGG
from nets.backbone.mobile_netv2 import mobilenet_v2, MobileNetV2
from .ssd_utils import *

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSD(nn.Module):
    def __init__(self, phase, base, extras, head, num_classes, confidence, nms_iou, backbone_name="vgg"):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = Config
        if backbone_name == "vgg":
            self.vgg = nn.ModuleList(base)
            self.L2Norm = L2Norm(512, 20)
        else:
            self.mobilenet = base
            self.L2Norm = L2Norm(96, 20)

        self.extras = nn.ModuleList(extras)
        self.priorbox = PriorBox(backbone_name, self.cfg)
        with torch.no_grad():
            self.priors = torch.tensor(self.priorbox.forward()).type(torch.float32)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.backbone_name = backbone_name
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, confidence, nms_iou)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # ---------------------------#
        #   获得conv4_3的内容
        #   shape为38,38,512
        # ---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23):
                x = self.vgg[k](x)
        else:
            for k in range(14):
                x = self.mobilenet[k](x)
        # ---------------------------#
        #   conv4_3的内容
        #   需要进行L2标准化
        # ---------------------------#
        s = self.L2Norm(x)
        sources.append(s)

        # ---------------------------#
        #   获得conv7的内容
        #   shape为19,19,1024
        # ---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)
        else:
            for k in range(14, len(self.mobilenet)):
                x = self.mobilenet[k](x)

        sources.append(x)
        # -------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        # -------------------------------------------------------------#
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if self.backbone_name == "vgg":
                if k % 2 == 1:
                    sources.append(x)
            else:
                sources.append(x)

        # -------------------------------------------------------------#
        #   为获得的6个有效特征层添加回归预测和分类预测
        # -------------------------------------------------------------#
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # -------------------------------------------------------------#
        #   进行reshape方便堆叠
        # -------------------------------------------------------------#
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # -------------------------------------------------------------#
        #   loc会reshape到batch_size,num_anchors,4
        #   conf会reshap到batch_size,num_anchors,self.num_classes
        #   如果用于预测的话，会添加上detect用于对先验框解码，获得预测结果
        #   不用于预测的话，直接返回网络的回归预测结果和分类预测结果用于训练
        # -------------------------------------------------------------#
        if self.phase == "test":
            output = self.detect.forward(
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                self.priors
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output
'''
还没结束！！！！！！！！！！！
'''
def get_vgg_ssd(phase, num_classes=21, confidence=0.5, nms_iou=0.45):

    mbox = [4, 6, 6, 6, 4, 4]
    vgg_backbone_list = VGG.get_vgg_backbone()
    extra_layer_list = VGG.get_extras()

    #下面是获得loc_layers 和 conf_layers
    backbone_source_index_list = [21, -2]
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(backbone_source_index_list):
        loc_layers += [nn.Conv2d(vgg_backbone_list[v].out_channels,
                                 mbox[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg_backbone_list[v].out_channels,
                                  mbox[k] * num_classes, kernel_size=3, padding=1)]
        # -------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        # -------------------------------------------------------------#
    for k, v in enumerate(extra_layer_list[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                  * num_classes, kernel_size=3, padding=1)]

    return SSD(phase, vgg_backbone_list, extra_layer_list, (loc_layers, conf_layers),
               num_classes, confidence, nms_iou, backbone_name="vgg")

def get_mobilenetV2_ssd(phase, num_classes=21, confidence=0.5, nms_iou=0.45):

    backbone, extra_layers = mobilenet_v2(pretrained=False).features, MobileNetV2.get_extra(1280)
    mbox = [6, 6, 6, 6, 6, 6]

    loc_layers = []
    conf_layers = []

    backbone_source = [13, -1]
    for k, v in enumerate(backbone_source):
        loc_layers += [nn.Conv2d(backbone[v].out_channels,
                                 mbox[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(backbone[v].out_channels,
                                  mbox[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers, 2):
        loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                  * num_classes, kernel_size=3, padding=1)]

    # -------------------------------------------------------------#
    #   add_vgg和add_extras，一共获得了6个有效特征层，shape分别为：
    #   (38,38,512), (19,19,1024), (10,10,512),
    #   (5,5,256), (3,3,256), (1,1,256)
    # -------------------------------------------------------------#
    SSD_MODEL = SSD(phase, backbone, extra_layers, (loc_layers, conf_layers), num_classes, confidence, nms_iou,
                    "mobilenet")
    return SSD_MODEL

