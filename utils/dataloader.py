from random import shuffle

from torch.utils.data import Dataset
import torch
import numpy as np
import cv2

from utils.data_processing import BaseDetectionDataProcessing

'''
1.在数据预处理部分代码是相同的   生成model_input是类似的，生成model_target是不同的
2.test 是做不了的，只能做train和validation
3.只需要重写 process_bboxes, model_input已经处理好了
'''
class AbstractDetectionDataset(Dataset):
    def __init__(self, train_lines, is_train, ddp: BaseDetectionDataProcessing):
        super(AbstractDetectionDataset, self).__init__()

        self.train_lines = train_lines # ['image_file_path.jpg 22,22,22,22,8 22,22,22,22,8','..',..]
        self.is_train = is_train       #True: train  False: test or validation
        self.ddp = ddp
        self.ddp.return_default = False  #必须要确保这一点
        self.input_shape = ddp.input_shape #(h,w,c) for Deep Nets

    def __len__(self):
        return len(self.train_lines)

    def _read_annotation_line(self, annotation_line: str):
        data_list = annotation_line.split()
        img = cv2.imread(data_list[0])
        bboxes = np.array([list(map(int, bbox_str.split(','))) for bbox_str in data_list[1:]])
        class_array = bboxes[:, -1]
        bboxes = bboxes[:, :-1]
        return img, bboxes, class_array

    def _train_step(self, annotation_line: str):
        img, bboxes, class_array = self._read_annotation_line(annotation_line)
        img, bboxes = self.ddp.train_step(img, bboxes) #返回值数据类型都是float32，无论输入是什么类型

        return img, bboxes, class_array.astype(np.float32)

    def _validation_step(self, annotation_line: str):
        img, bboxes, class_array = self._read_annotation_line(annotation_line)
        img, bboxes = self.ddp.validation_step(img, bboxes)  # 返回值数据类型都是float32，无论输入是什么类型

        return img, bboxes, class_array.astype(np.float32)

    def process_bboxes(self, img: np.ndarray, bboxes: np.ndarray, class_array: np.ndarray):
        '''
        都是float32类型
        '''
        raise NotImplementedError

    def __getitem__(self, item):
        if item == 0:
            shuffle(self.train_lines)
        annotation_line = self.train_lines[item]
        if self.is_train:
            img, bboxes, class_array = self._train_step(annotation_line)
        else:
            img, bboxes, class_array = self._validation_step(annotation_line)

        return self.process_bboxes(img, bboxes, class_array)


class SSDDataset(AbstractDetectionDataset):

    def __init__(self, train_lines, is_train, ddp: BaseDetectionDataProcessing):

        super(SSDDataset, self).__init__(train_lines, is_train, ddp)

    def process_bboxes(self, img: np.ndarray, bboxes: np.ndarray, class_array: np.ndarray):
        #只需要归一化bboxes，转换img维度
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / self.input_shape[1]
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / self.input_shape[0]
        bboxes = np.maximum(np.minimum(bboxes, 1.), 0.) #确保可靠
        n = bboxes.shape[0]
        new_bboxes = np.empty((n, 5), dtype=np.float32)
        new_bboxes[:, :-1] = bboxes
        new_bboxes[:, -1] = class_array
        img = np.transpose(img, (2, 0, 1))
        return img, new_bboxes   #(3, h, w)  (n, 5)

def torch_ssd_collate_fn(batch):
    '''
    batch: [(第1个样本),(第2个样本)]
    返回的是tensor(bsz,3,h,w)和List(tensor(*,5),....)
    '''

    img = []
    all_bboxes = []
    for img_i, all_bboxes_i in batch:
        img.append(img_i)
        all_bboxes.append(torch.tensor(all_bboxes_i))

    return torch.from_numpy(np.array(img, dtype=np.float32)), all_bboxes
