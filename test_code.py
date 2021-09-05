import cv2
import numpy as np

from utils.data_processing import BaseDetectionDataProcessing
from utils.dataloader import torch_ssd_collate_fn
from utils.utils import show_pic
'''
************************************
A.1: 以下为测试DetectionDataProcessing
************************************
'''
# annotation_path = '2007_train.txt'
#
# img = cv2.imread(r'D:\Project\CV\yolo3_learning/VOCdevkit/VOC2007/JPEGImages/000005.jpg')
# bboxes = np.array([[263,211,324,339], [165,264,253,372], [241,194,295,299]])
#
# show_pic(img, bboxes, name='111')
# print('bboxes',bboxes)
# ddp = BaseDetectionDataProcessing((512,512,3), return_default=True)
# new_img, new_bboxes = ddp.change_size(img, bboxes)
# new_img = new_img.astype(img.dtype)
#
# new_bboxes = np.ceil(new_bboxes).astype(np.int32)
# print('new_bboxes', new_bboxes)
# show_pic(new_img, new_bboxes, name='222')
#
#
# new_img, new_bboxes, dx, dy, iw, ih, nw, nh = ddp.add_letterbox(img, bboxes)
# new_img = new_img.astype(img.dtype)
#
# new_bboxes = np.ceil(new_bboxes).astype(np.int32)
# print('new_bboxes', new_bboxes)
# show_pic(new_img, new_bboxes, name='222')
#
# old_bboxes = ddp.decode_letterbox(new_bboxes, dx, dy, iw, ih, nw, nh)
# print('old bboxes: ', old_bboxes)

#总接口 train_step
# for i in range(10):
#     i_img = img.copy()
#     new_img, new_bboxes = ddp.validation_step(i_img, bboxes)
#     show_pic(new_img, new_bboxes, name='222')
'''
************************************
A.2 以下为测试Dataset
************************************
'''
# from utils.dataloader import SSDDataset
# test_num = 2
# with open('2007_train.txt', 'r', encoding='utf-8') as file:
#     train_lines = file.readlines()[:test_num]

##没有normalize image
# ddp = BaseDetectionDataProcessing((300,300,3), return_default=False, is_normal_image=False)
# ssd_dataset = SSDDataset(train_lines, is_train=True, ddp=ddp)
# for i in range(test_num):
#     img, bboxes, class_array = ssd_dataset[i]
#     img = img.astype(np.uint8)
#     bboxes = bboxes.astype(np.int32)
#     print(i, ': ', class_array)
#     show_pic(img, bboxes)
## 已经normalize_image 了, return_default=False, is_normal_image=True
# ddp = BaseDetectionDataProcessing((300,300,3))
# ssd_dataset = SSDDataset(train_lines, is_train=False, ddp=ddp)
# for i in range(test_num):
#     img, all_bboxes = ssd_dataset[i]
#     print(img.shape,all_bboxes.shape)
# from torch.utils.data import DataLoader
# ssd_dataloader = DataLoader(ssd_dataset, batch_size=5 ,shuffle=True, collate_fn=torch_ssd_collate_fn)
# for i, batch in enumerate(ssd_dataloader):
#     print('The {}th'.format(i))
    # img, all_bboxes = batch
    # print(img.shape, all_bboxes[0].shape)



'''
************************************
B.1 以下测试网络构建部分
************************************
'''
from torchsummary import summary
from nets.backbone.vgg import VGG
from nets.backbone.mobile_netv2 import MobileNetV2
import torch

# from nets.ssd.ssd_net import *
# model = get_mobilenetV2_sdd("test")
# state_dict = torch.load("model_data/mobilenetv2_ssd_weights.pth")
# model.load_state_dict(state_dict)
# net = model.cuda()
# net = VGG().cuda()
# print(len(net.model))
# for i in net.index_list:
#     print(net.model[i])
# summary(net, input_size=[(3, 300, 300)], batch_size=8) #[bsz, 1024, 19, 19]
# print(len(list(net.parameters())))
# net = MobileNetV2(pretrained=False).cuda()
# summary(net, input_size=[(3, 300, 300)], batch_size=8) #[bsz, 96, 19, 19]
# print(len(list(net.parameters())))
'''
!!!!!!!!!!!!!!!
以下为模型训练
!!!!!!!!!!!!!!!
'''
from nets.ssd.ssd import SSDTorchTrainDetectionNet, SSDTorchDetectionNet
from nets.ssd.ssd_utils import SSDDataset
from nets.torch_net import TorchTrainer
from nets.ssd.ssd_net import get_mobilenetV2_ssd
from utils.data_processing import BaseDetectionDataProcessing
# '''
# 数据准备
# '''
# ddp = BaseDetectionDataProcessing(input_shape=(300,300,3),is_add_letterbox=False)
# val_split = 0.2
# annotation_path = '2007_train.txt'
# with open(annotation_path) as f:
#     lines = f.readlines()
# np.random.seed(10101)
# np.random.shuffle(lines)
# np.random.seed(None)
# num_val = int(len(lines)*val_split)
# num_train = len(lines) - num_val
# train_dataset = SSDDataset(lines[:num_train], is_train=True,ddp=ddp)
# val_dataset = SSDDataset(lines[num_train:], is_train=True,ddp=ddp)
# '''
# 训练模型
# '''
# model_data_path = 'model_data/mobilenetv2_ssd_weights.pth'
# torch_net = get_mobilenetV2_ssd('train')
# torch_net.load_state_dict(torch.load(model_data_path))
# ssd_detection_net = SSDTorchTrainDetectionNet([train_dataset, val_dataset],5,torch_net,None)
# trainer = TorchTrainer(ssd_detection_net, 10)
# trainer.fit()

'''
以下为测试
'''
# from nets.ssd import SSDDetectionFactory
# ssd_detection_net = SSDDetectionFactory().get_detection_net()
#
# # from PIL import Image
# # img = Image.open('img/street.jpg')
# img = cv2.imread('img/csgo.jpg')
# # cv2.imshow('kkk',img)
# # cv2.waitKey(0)
# result_img = ssd_detection_net.test_step(img)
# cv2.imwrite('img/test.jpg', result_img)

'''
！！！！！！！！！！！！！！！！！！
Final Result
！！！！！！！！！！！！！！！！！！
'''
from nets.ssd import SSDDetectionFactory
factory = SSDDetectionFactory()
ssd_detection_net = factory.get_detection_net()

img = cv2.imread('img/csgo.jpg')
ssd_detection_net.test_step(img)

ssd_trainer = factory.get_detection_trainer()
ssd_trainer.fit()