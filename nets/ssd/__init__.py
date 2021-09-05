import numpy as np
import torch

from .ssd import SSDTorchDetectionNet, SSDTorchTrainDetectionNet
from .ssd_net import get_mobilenetV2_ssd, get_vgg_ssd
from .ssd_utils import SSDDataset, SSDDetectionDataProcessing
from nets.basenet import BaseDetectionNetFactory
from nets.torch_net import TorchTrainer
from utils.data_processing import BaseDetectionDataProcessing


class SSDDetectionFactory(BaseDetectionNetFactory):

    def get_detection_net(self, *args, **kwargs):
        #input_shape is fixed
        ddp = SSDDetectionDataProcessing(input_shape=(300, 300, 3), **kwargs)
        model = SSDTorchDetectionNet(ddp)
        return model

    def get_detection_trainer(self, *args, annotation_path = '2007_train.txt', val_split = 0.2,
                              model_data_path = 'model_data/mobilenetv2_ssd_weights.pth', batch_size=5, max_epoch=10,
                              device_name='cuda:0'):
        #data processing  input_shape is fixed
        ddp = SSDDetectionDataProcessing(input_shape=(300,300,3),is_add_letterbox=False)

        #data preparation
        with open(annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines)*val_split)
        num_train = len(lines) - num_val
        train_dataset = SSDDataset(lines[:num_train], is_train=True,ddp=ddp)
        val_dataset = SSDDataset(lines[num_train:], is_train=True,ddp=ddp)

        #init Trainer
        torch_net = get_mobilenetV2_ssd('train')
        torch_net.load_state_dict(torch.load(model_data_path))
        ssd_detection_net = SSDTorchTrainDetectionNet([train_dataset, val_dataset], batch_size, torch_net)
        trainer = TorchTrainer(ssd_detection_net, max_epoch)

        return trainer