import os

import numpy as np
'''
For model:
All the detection model needs to generalize BaseDetectionNet, BaseDetectionTrainer
'''
BASE_CONFIG = {
    'class_path' : '',
}

class BaseDetectionNet(object):

    def __init__(self, ddp, config):
        super(BaseDetectionNet, self).__init__()
        self.ddp = ddp
        self._config = config
        self.class_names = self._get_class() #list
        self.model = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.get('classes_path'))
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    def get(self, key):
        return self._config.get(key)

    def generate(self):
        '''
        每一个框架有不同的载入模型的方式
        return model  不需要管放在哪个设备
        '''
        raise NotImplementedError

    def adapt_model_input(self, model_input):
        '''
        转化为torch.tensor 或者其他框架的模型输入格式
        model_input: (bsz,h,w,c)
        以torch为例子
        return torch.tensor(model_input, dtype=torch.float32, device=..)
        '''
        pass

    def forward(self, x):
        '''
        Adapting for self.model
        if self.model is instance of torch.nn.Module then  return self.model(x)
        if self.model is instance of keras.Model then ....

        return for self.decode_model_output(model_output)
        '''
        raise NotImplementedError

    def decode_model_output(self, model_output):
        #return final bboxes
        raise NotImplementedError

    def test_end(self, raw_img, bboxes, bboxes_info):
        '''
        return for self.test_step
        usually you need to draw bboxes and bboxes information on the raw_img
        '''
        raise NotImplementedError

    def test_step(self, raw_img):
        '''
        本接口只支持  一张图片！
        raw_img.shape is (h, w, c)
        '''
        new_img = self.ddp.test_step(raw_img)    #raw_img 建议为cv2.imread 得到的，且BGR为格式
        model_input = np.expand_dims(new_img, 0)  # np.array(1,h,w,c) usually float32
        model_input = self.adapt_model_input(model_input)
        model_output = self.forward(model_input)
        bboxes, bboxes_info = self.decode_model_output(model_output)  #如果需要raw_img相关信息，在ddp中设置
        #bboxes must be numpy.ndarray and info of bboxes. recommand bboxes_info=[box_conf, box_label]
        return self.test_end(raw_img, bboxes, bboxes_info)

class BaseDetectionTrainer(object):

    def __init__(self):
        '''
        建议组合  深度学习模型
        '''
        super(BaseDetectionTrainer, self).__init__()

    def fit(self):
        pass

class BaseDetectionNetFactory(object):

    def get_detection_net(self, *args, **kwargs):
        raise NotImplementedError

    def get_detection_trainer(self, *args, **kwargs):
        raise NotImplementedError