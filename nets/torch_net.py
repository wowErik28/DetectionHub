import os

import torch
from tqdm import tqdm

from .basenet import BaseDetectionNet, BaseDetectionTrainer

Torch_Config = {
    "model_path"        : 'model_data/mobilenetv2_ssd_weights.pth',
    "classes_path"      : 'model_data/voc_classes.txt',
    "input_shape"       : (300, 300, 3),
    "confidence"        : 0.5,
    "nms_iou"           : 0.45,
    "device_name"       : 'cuda:0',
    #-------------------------------#
    #   主干网络的选择
    #   vgg或者mobilenet
    #-------------------------------#
    "backbone"          : "mobilenetv2",
}

class BaseTorchDetectionNet(BaseDetectionNet):

    def __init__(self, ddp, config):
        super(BaseTorchDetectionNet, self).__init__(ddp, config)
        self.device = torch.device(self.get('device_name'))
        self.set_model_to_device()

    def set_model_to_device(self):
        self.model = self.model.to(self.device)
    def generate(self):
        raise NotImplementedError

    def adapt_model_input(self, model_input):
        return torch.tensor(model_input, dtype=torch.float32, device=self.device)

    def forward(self, x):

        with torch.no_grad():

            return self.model(x)

    def decode_model_output(self, model_output):
        #返回最终的 bbox
        raise NotImplementedError

    def test_end(self, raw_img, bboxes, bboxes_info):
        raise NotImplementedError


class BaseTorchTrainDetectionNet(object):

    def __init__(self, dataset:list, batch_size, deepnet: torch.nn.Module, state_dict_path=None):
        super(BaseTorchTrainDetectionNet, self).__init__()
        #prepare data for training
        self.train_dataset, self.val_dataset = dataset
        self.batch_size = batch_size
        self.train_dataloader = self.get_train_dataloader()
        self.val_dataloader = self.get_val_dataloader()

        #prepare model
        self.model = deepnet
        if state_dict_path is not None:
            print('Loading state dict for training.......')
            self.model.load_state_dict(state_dict_path)

    def forward(self, x:torch.Tensor):
        return self.model(x)

    def init_logger(self):
        '''
        定义需要保存的指标
        '''
        return {'loss':[],'val_loss':[]}

    def get_train_dataloader(self):
        pass
    def get_val_dataloader(self):
        pass
    def batch_to_device(self, batch, device):
        raise NotImplementedError

    def train_step(self, batch, batch_idx):
        '''
        This interface cannot be realized here since we do not konw how to unpack batch
        '''
        raise NotImplementedError
    def val_step(self, batch, batch_idx):
        return None
    def train_step_end(self, train_step_outputs)->dict:
        raise NotImplementedError
    def val_step_end(self, val_step_outputs)->dict:
        raise NotImplementedError
    '''
    用于log、保存checkpoint、earlystopping等callbacks
    '''
    def val_epoch_end(self, epoch_idx, val_step_outputs, logger):
        '''
        val_step_outputs: [val_step_output1, val_step_output2, .....]
        '''
        pass

    def train_epoch_end(self, epoch_idx, train_step_outputs, logger):
        '''
        train_step_outputs:[train_step_output1,train_step_output2,....]
        '''
        pass

class TorchTrainer(BaseDetectionTrainer):
    '''
    这里管理device   epochs
    '''
    def __init__(self, deepnet:BaseTorchTrainDetectionNet, max_epoch, device_name='cuda:0'):
        super(TorchTrainer, self).__init__()
        self.model = deepnet
        self.device = torch.device(device_name)
        self.model.model.to(self.device)

        self.max_epoch = max_epoch
        self.configure_epoch_info()

        self.logger = self.model.init_logger()

    def configure_epoch_info(self):
        batch_size = self.model.batch_size
        self.epoch_size = len(self.model.train_dataset) // batch_size
        self.epoch_size_val = len(self.model.val_dataset) // batch_size

    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch

    def fit(self):

        for epoch_idx in range(self.max_epoch):
            self.fit_one_epoch(epoch_idx=epoch_idx)

    def fit_one_epoch(self, epoch_idx):

        self.model.model.train()
        train_step_outputs = [] #store the batch info
        with tqdm(total=self.epoch_size, desc=f'Train Epoch {epoch_idx + 1}/{self.max_epoch}',
                  postfix=dict, mininterval=0.3) as pbar:
            for batch_idx, batch in enumerate(self.model.train_dataloader):
                batch = self.model.batch_to_device(batch, self.device)
                train_step_output = self.model.train_step(batch, batch_idx)
                train_step_outputs.append(train_step_output)
                pbar_info = self.model.train_step_end(train_step_outputs)
                pbar.set_postfix(**pbar_info)
                pbar.update(1)
            #handel train_epoch_list
            self.model.train_epoch_end(epoch_idx, train_step_outputs, self.logger)


        self.model.model.eval()
        val_step_outputs = []
        with tqdm(total=self.epoch_size_val, desc=f'Valid Epoch {epoch_idx + 1}/{self.max_epoch}',
                  postfix=dict, mininterval=0.3) as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.model.val_dataloader):
                    batch = self.model.batch_to_device(batch, self.device)
                    val_step_output = self.model.val_step(batch, batch_idx)
                    val_step_outputs.append(val_step_output)
                    pbar_info = self.model.val_step_end(val_step_outputs)
                    pbar.set_postfix(**pbar_info)
                    pbar.update(1)
                # handel val_epoch_list
                self.model.val_epoch_end(epoch_idx, val_step_outputs, self.logger)

class TorchCallBack(object):
    def __init__(self):
        super(TorchCallBack, self).__init__()

    def run(self, val_info_dict):
        pass

