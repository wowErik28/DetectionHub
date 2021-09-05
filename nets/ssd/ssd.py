import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2

from nets.ssd import ssd_net
from nets.ssd.config import SSD_Default_Config,Config
from nets.torch_net import BaseTorchDetectionNet, BaseTorchTrainDetectionNet
from nets.ssd.ssd_utils import torch_ssd_collate_fn, MultiBoxLoss


# SSD_Default_Config = {
#     "model_path"        : 'model_data/mobilenetv2_ssd_weights.pth',
#     "classes_path"      : 'model_data/voc_classes.txt',
#     "input_shape"       : (300, 300, 3),
#     "confidence"        : 0.5,
#     "nms_iou"           : 0.45,
#     "device_name"       : 'cuda:0',
#     #-------------------------------#
#     #   主干网络的选择
#     #   vgg或者mobilenet
#     #-------------------------------#
#     "backbone"          : "mobilenetV2",
#     # "model_path"        : 'model_data/ssd_weights.pth',
#     # "backbone"          : "vgg",
#     "add_letterbox"     : False,
# }

class SSDTorchDetectionNet(BaseTorchDetectionNet):

    def __init__(self, ddp, config = SSD_Default_Config):
        super(SSDTorchDetectionNet, self).__init__(ddp, config)

    def generate(self):
        self.num_classes = len(self.class_names) + 1
        self.whole_model_name = 'get_{}_ssd'.format(self.get('backbone'))
        model = getattr(ssd_net, self.whole_model_name)("test", num_classes=self.num_classes,
                                                        confidence=self.get('confidence'),
                                                        nms_iou=self.get('nms_iou'))
        print('Loading weights into state dict...')
        model.load_state_dict(torch.load(self.get('model_path')))
        model = model.eval()
        print('Finish Loading')
        return model

    def decode_model_output(self, model_output):
        # 返回最终的 bbox
        preds = model_output
        top_conf = []
        top_label = []
        top_bboxes = []
        # ---------------------------------------------------#
        #   preds的shape为 1, num_classes, top_k, 5
        # ---------------------------------------------------#
        for i in range(preds.size(1)):
            j = 0
            while preds[0, i, j, 0] >= self.get('confidence'):
                # ---------------------------------------------------#
                #   score为当前预测框的得分
                #   label_name为预测框的种类
                # ---------------------------------------------------#
                score = preds[0, i, j, 0]
                label_name = self.class_names[i - 1]
                # ---------------------------------------------------#
                #   pt的shape为4, 当前预测框的左上角右下角
                # ---------------------------------------------------#
                pt = (preds[0, i, j, 1:]).detach().numpy()
                coords = [pt[0], pt[1], pt[2], pt[3]]
                top_conf.append(score)
                top_label.append(label_name)
                top_bboxes.append(coords)
                j = j + 1

        # 如果不存在满足门限的预测框，直接返回原图

        if len(top_conf) <= 0:
            return None, None

        top_conf = np.array(top_conf) #()
        top_label = np.array(top_label)
        top_bboxes = np.array(top_bboxes)

        if self.get("add_letterbox"):
            boxes = self.ddp.decode_letterbox(top_bboxes, *self.ddp.test_letterbox_info)
        else:
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
            top_xmin = top_xmin * self.ddp.raw_img_shape[1]
            top_ymin = top_ymin * self.ddp.raw_img_shape[0]
            top_xmax = top_xmax * self.ddp.raw_img_shape[1]
            top_ymax = top_ymax * self.ddp.raw_img_shape[0]
            boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

        return boxes, [top_conf, top_label]

    def test_end(self, raw_img, bboxes, bboxes_info):
        if bboxes is None or bboxes_info is None:
            print('nothing find')
            return None
        top_conf, top_label = bboxes_info
        boxes = bboxes
        raw_img = raw_img.copy()
        for i, c in enumerate(top_label):
            predicted_class = c
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(raw_img)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(raw_img)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)

            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            raw_img = cv2.rectangle(raw_img, (left, top),(right,bottom),(0,0,255),8)
            # cv2.putText(raw_img,label)
            cv2.putText(raw_img, label.decode('utf-8'), (left, top+20),
                        cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),5)
        cv2.imshow('img', raw_img)
        cv2.waitKey(0)

        return  raw_img

class SSDTorchTrainDetectionNet(BaseTorchTrainDetectionNet):

    def __init__(self, dataset:list, batch_size, deepnet: torch.nn.Module, state_dict_path=None, lr=5e-4):
        super(SSDTorchTrainDetectionNet, self).__init__(dataset, batch_size, deepnet, state_dict_path)
        self.criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5, False, True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    def init_logger(self):
        return {'loc_loss':[],'conf_loss':[],'val_loc_loss':[],'val_conf_loss':[]}

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          collate_fn= torch_ssd_collate_fn, drop_last=True, shuffle=True)
    def get_val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn= torch_ssd_collate_fn, drop_last=True)

    def batch_to_device(self, batch, device):
        images, targets = batch[0], batch[1]
        images = images.to(device)
        targets = [ann.to(device) for ann in targets]
        return (images, targets)

    def train_step(self, batch, batch_idx):
        images, targets = batch[0], batch[1]
        out = self.model(images)
        # ----------------------#
        #   清零梯度
        # ----------------------#
        self.optimizer.zero_grad()
        # ----------------------#
        #   计算损失
        # ----------------------#
        loss_l, loss_c = self.criterion(out, targets)
        loss = loss_l + loss_c
        # ----------------------#
        #   反向传播
        # ----------------------#
        loss.backward()
        self.optimizer.step()

        loc_loss = loss_l.item()
        conf_loss = loss_c.item()

        return loc_loss, conf_loss

    def val_step(self, batch, batch_idx):
        images, targets = batch[0], batch[1]
        out = self.model(images)
        # ----------------------#
        #   清零梯度
        # ----------------------#
        self.optimizer.zero_grad()
        # ----------------------#
        #   计算损失
        # ----------------------#
        loss_l, loss_c = self.criterion(out, targets)

        loc_loss = loss_l.item()
        conf_loss = loss_c.item()

        return loc_loss, conf_loss
    def train_step_end(self, train_step_outputs):
        loc_loss = [output[0] for output in train_step_outputs]
        conf_loss = [output[1] for output in train_step_outputs]
        iteration = len(train_step_outputs)
        loc_loss = sum(loc_loss) / iteration
        conf_loss = sum(conf_loss) / iteration
        return {'loc_loss':loc_loss, 'conf_loss':conf_loss}

    def val_step_end(self, val_step_outputs):
        loc_loss = [output[0] for output in val_step_outputs]
        conf_loss = [output[1] for output in val_step_outputs]
        iteration = len(val_step_outputs)
        loc_loss = sum(loc_loss) / iteration
        conf_loss = sum(conf_loss) / iteration
        return {'val_loc_loss': loc_loss, 'val_conf_loss': conf_loss}

    def train_epoch_end(self, epoch_idx, train_step_outputs, logger):
        '''
        train_step_outputs:[train_step_output1,train_step_output2,....]
        '''
        loc_loss = [output[0] for output in train_step_outputs]
        conf_loss = [output[1] for output in train_step_outputs]
        loc_loss = sum(loc_loss) / len(loc_loss)
        conf_loss = sum(conf_loss) / len(conf_loss)
        logger['loc_loss'].append(loc_loss)
        logger['conf_loss'].append(conf_loss)



    def val_epoch_end(self, epoch_idx, val_step_outputs, logger):
        '''
        val_step_outputs: [val_step_output1, val_step_output2, .....]
        '''
        loc_loss = [output[0] for output in val_step_outputs]
        conf_loss = [output[1] for output in val_step_outputs]
        loc_loss = sum(loc_loss) / len(loc_loss)
        conf_loss = sum(conf_loss) / len(conf_loss)
        logger['val_loc_loss'].append(loc_loss)
        logger['val_conf_loss'].append(conf_loss)
        torch.save(self.model.state_dict(), 'logs/Epoch{} val_loc_loss:{:.5f} val_conf_loss'.format(
            epoch_idx+1,loc_loss,conf_loss))

