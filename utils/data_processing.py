'''
1.参数类型限制效果不佳，typing module is not friendly while using
2.如果要新加入Data Processing方法，继承BaseDetectionDataProcessing，加入新的function，然后test_step, train_step可能也会要改
'''
import torch
import cv2
from PIL import Image
import numpy as np

from utils.utils import decode_letterbox
class BaseDetectionDataProcessing:

    def __init__(self, input_shape, normal_rate=0.5, jitter=0.3, least_scale=0.25,
                 hue=.1, sat=1.5, val=1.5, flip_rate=0.5,
                 func_name_list=['change_hsv', 'flip_image', 'change_size'],
                 return_default = False, is_normal_image = True, channel_first=True,
                 is_add_letterbox=False, is_cv2_image=True):
        '''
        存各项参数
        input_shape: the input shape of Deep Net
        1.每一个函数都是默认接收和返回的img bboxes都是float32类型, 返回都是float32类型。所以在train_step,..必须转换类型，函数中不转换
           return_default可以变成 uint8 和 int32
        '''
        self.input_shape = input_shape
        self.h = input_shape[0]
        self.w = input_shape[1]

        #用于change_size 如果不想加letterbox 则直接让normal_rate=1.
        self.normal_rate = normal_rate #直接resize的概率
        self.jitter = jitter
        self.least_scale = least_scale
        #HSV转换
        self.hue = hue
        self.sat = sat
        self.val = val
        #用于flip
        self.flip_rate = flip_rate
        #用于train_step
        self.func_name_list = func_name_list
        self.return_default = return_default #是否返回img uint8和 bboxes int32
        #用于归一化图像, train和validation时候建议使用
        self.is_normal_image = is_normal_image
        #返回的图片为 (c,h,w)
        self.channel_first = channel_first
        self.is_add_letterbox = is_add_letterbox
        self.is_cv2_image = is_cv2_image
    def _rand(self, a=0., b=1.):
        assert (b > a)
        return np.random.rand() * (b - a) + a

    def change_hsv(self, img: np.ndarray, bboxes=None):
        #img 推荐是float32
        hue = self._rand(-self.hue, self.hue)
        sat = self._rand(1, self.sat) if self._rand() < .5 else 1 / self._rand(1, self.sat)
        val = self._rand(1, self.val) if self._rand() < .5 else 1 / self._rand(1, self.val)
        x = cv2.cvtColor(img / 255., cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        return image_data, bboxes


    def normalize_image(self, img: np.ndarray):
        MEANS = np.array([104, 117, 123])
        # mean = [0.40789655, 0.44719303, 0.47026116]
        # std = [0.2886383, 0.27408165, 0.27809834]
        # return ((np.float32(img) / 255.) - mean) / std
        return np.float32(img) - MEANS

    def change_size(self, img: np.ndarray, bboxes: np.ndarray):
        '''
        先调整 h与w的比例 然后缩放
        '''
        ih, iw, _ = img.shape

        if self._rand(0, 1) < self.normal_rate:
            #直接resize
            img = cv2.resize(img, (self.w, self.h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / iw * self.w
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / ih * self.h

            return img, bboxes

        new_hw_scale = (ih / iw) * self._rand(1 - self.jitter, 1 + self.jitter)
        nw = iw
        nh = nw * new_hw_scale

        img_scale = min(self.w / nw, self.h / nh)

        if img_scale > self.least_scale:
            img_scale = self._rand(self.least_scale, img_scale)
        
        nw = int(nw * img_scale)
        nh = int(nh * img_scale)
        img = cv2.resize(img, (nw, nh))

        dx = (self.w - nw) // 2
        dy = (self.h - nh) // 2
        new_img = np.zeros(self.input_shape, dtype=np.float32)
        new_img[dy: dy+nh, dx: dx+nw] = img.astype(np.float32)

        '''
        缩放过程中不能变的是比例
        '''
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / iw * nw + dx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / ih * nh + dy

        return new_img, bboxes

    def flip_image(self, img: np.ndarray, bboxes: np.ndarray):
        img = cv2.flip(img, 1)

        ih, iw, _ = img.shape

        bboxes[:, [0, 2]] = iw - bboxes[:, [2, 0]]

        return img, bboxes

    def train_step(self, img: np.ndarray, bboxes: np.ndarray):
        '''
        训练时调用这个接口
        必须要包含change_size(默认在最后)
        单独调用flip, change_hsv等是不行的
        默认返回类型都是float32
        '''
        if self.is_cv2_image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        bboxes = bboxes.astype(np.float32)
        for func_name in self.func_name_list:
            F = getattr(self, func_name)
            img, bboxes = F(img, bboxes)
        if self.is_normal_image:
            img = self.normalize_image(img)
            return img, bboxes

        if self.return_default:
            img = img.astype(np.uint8)
            bboxes = bboxes.astype(np.int32)
        if self.channel_first:
            img = np.transpose(img, (2,0,1))
        return img, bboxes

    def validation_step(self, img:np.ndarray, bboxes: np.ndarray):

        if self.is_cv2_image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        bboxes = bboxes.astype(np.float32)

        new_img, bboxes = self.add_letterbox(img, bboxes)

        if self.is_normal_image:
            new_img = self.normalize_image(new_img)
            return new_img, bboxes

        if self.return_default:
            new_img = new_img.astype(np.uint8)
            bboxes = bboxes.astype(np.int32)
        if self.channel_first:
            new_img = np.transpose(new_img, (2,0,1))
        return new_img, bboxes

    def add_letterbox(self, img: np.ndarray, bboxes: np.ndarray):
        '''
        该方法为validation 和 train阶段使用
        输出的img的shape为input_shape
        '''

        new_img, dx, dy, iw, ih, nw, nh = self.add_test_letterbox(img)


        #改变bboxes   bboxes首先是变化了大小 然后是平移
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / iw * nw + dx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / ih * nh + dy

        return new_img, bboxes

    def add_test_letterbox(self, img: np.ndarray):
        '''
        输出的img的shape为input_shape
        返回为float32
        '''
        # img = img.copy()
        ih, iw, _ = img.shape
        # 确定img_scale
        img_scale = min(self.w / iw, self.h / ih)
        # 确定nh nw dx dy
        nw = int(iw * img_scale)
        nh = int(ih * img_scale)
        dx = (self.w - nw) // 2
        dy = (self.h - nh) // 2

        # 赋值new_img
        img = cv2.resize(img, (nw, nh),interpolation=cv2.INTER_CUBIC)
        new_img = np.zeros(self.input_shape, dtype=np.float32) + 128
        new_img[dy: dy + nh, dx: dx + nw] = img


        return new_img, dx, dy, iw, ih, nw, nh

    def decode_letterbox(self, output_bboxes: np.ndarray, dx, dy, iw, ih, nw, nh):
        return decode_letterbox(output_bboxes, dx, dy, iw, ih, nw, nh)

    def test_step(self, img: np.ndarray):
        '''
        必须要img.copy() 不能破坏原图
        默认返回类型为float32
        '''
        img = img.copy()
        if self.is_cv2_image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_img = None
        if self.is_add_letterbox:
            new_img, dx, dy, iw, ih, nw, nh = self.add_test_letterbox(img)
            self.test_letterbox_info = [dx, dy, iw, ih, nw, nh]
        else:
            new_img = cv2.resize(img, (self.w, self.h),interpolation=cv2.INTER_CUBIC)
            self.raw_img_shape = img.shape
        new_img = new_img.astype(np.float32)
        if self.is_normal_image:
            new_img = self.normalize_image(new_img)

        if self.channel_first:
            new_img = np.transpose(new_img, (2,0,1))

        return new_img

    # def test_step(self, image):
    #     '''
    #     必须要img.copy() 不能破坏原图
    #     默认返回类型为float32
    #     '''
    #     image = image.convert('RGB')
    #
    #     image_shape = np.array(np.shape(image)[0:2])
    #     self.raw_img_shape = image_shape
    #     # ---------------------------------------------------------#
    #     #   给图像增加灰条，实现不失真的resize
    #     #   也可以直接resize进行识别
    #     # ---------------------------------------------------------#
    #
    #     crop_img = image.resize((self.input_shape[1], self.input_shape[0]), Image.BICUBIC)
    #
    #     photo = np.array(crop_img, dtype=np.float64)
    #     MEANS = np.array([104, 117, 123])
    #     photo = np.transpose(photo - MEANS, (2, 0, 1))
    #     return photo
