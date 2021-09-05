from math import sqrt as sqrt

import numpy as np
import torch
from PIL import Image
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

from nets.ssd.config import Config
from utils.data_processing import BaseDetectionDataProcessing
from utils.dataloader import AbstractDetectionDataset

class SSDDetectionDataProcessing(BaseDetectionDataProcessing):
    def __init__(self, input_shape=(300,300,3),is_add_letterbox=False):
        super(SSDDetectionDataProcessing, self).__init__(input_shape, is_add_letterbox)

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

class Detect(Function):
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = Config['variance']

    def forward(self, loc_data, conf_data, prior_data):
        # --------------------------------#
        #   先转换成cpu下运行
        # --------------------------------#
        loc_data = loc_data.cpu()
        conf_data = conf_data.cpu()

        # --------------------------------#
        #   num的值为batch_size
        #   num_priors为先验框的数量
        # --------------------------------#
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        # --------------------------------------#
        #   对分类预测结果进行reshape
        #   num, num_classes, num_priors
        # --------------------------------------#
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # 对每一张图片进行处理正常预测的时候只有一张图片，所以只会循环一次
        for i in range(num):
            # --------------------------------------#
            #   对先验框解码获得预测框
            #   解码后，获得的结果的shape为
            #   num_priors, 4
            # --------------------------------------#
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()

            # --------------------------------------#
            #   获得每一个类对应的分类结果
            #   num_priors,
            # --------------------------------------#
            for cl in range(1, self.num_classes):
                # --------------------------------------#
                #   首先利用门限进行判断
                #   然后取出满足门限的得分
                # --------------------------------------#
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # --------------------------------------#
                #   将满足门限的预测框取出来
                # --------------------------------------#
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # --------------------------------------#
                #   利用这些预测框进行非极大抑制
                # --------------------------------------#
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output


class PriorBox(object):
    def __init__(self, backbone_name, cfg):
        super(PriorBox, self).__init__()
        # 获得输入图片的大小，默认为300x300
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps'][backbone_name]
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = [cfg['min_dim'] / x for x in cfg['feature_maps'][backbone_name]]
        self.aspect_ratios = cfg['aspect_ratios'][backbone_name]
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # ----------------------------------------#
        #   对feature_maps进行循环
        #   利用SSD会获得6个有效特征层用于预测
        #   边长分别为[38, 19, 10, 5, 3, 1]
        # ----------------------------------------#
        for k, f in enumerate(self.feature_maps):
            # ----------------------------------------#
            #   分别对6个有效特征层建立网格
            #   [38, 19, 10, 5, 3, 1]
            # ----------------------------------------#
            x, y = np.meshgrid(np.arange(f), np.arange(f))
            x = x.reshape(-1)
            y = y.reshape(-1)
            # ----------------------------------------#
            #   所有先验框均为归一化的形式
            #   即在0-1之间
            # ----------------------------------------#
            for i, j in zip(y, x):
                f_k = self.image_size / self.steps[k]
                # ----------------------------------------#
                #   计算网格的中心
                # ----------------------------------------#
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # ----------------------------------------#
                #   获得小的正方形
                # ----------------------------------------#
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # ----------------------------------------#
                #   获得大的正方形
                # ----------------------------------------#
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # ----------------------------------------#
                #   获得两个的长方形
                # ----------------------------------------#
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # ----------------------------------------#
        #   获得所有的先验框 8732,4
        # ----------------------------------------#
        output = torch.Tensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)
        return output

def point_form(boxes):
    # ------------------------------#
    #   获得框的左上角和右下角
    # ------------------------------#
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)


def center_size(boxes):
    # ------------------------------#
    #   获得框的中心和宽高
    # ------------------------------#
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2,
                     boxes[:, 2:] - boxes[:, :2], 1)


def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    # ------------------------------#
    #   获得交矩形的左上角
    # ------------------------------#
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    # ------------------------------#
    #   获得交矩形的右下角
    # ------------------------------#
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)
    # -------------------------------------#
    #   计算先验框和所有真实框的重合面积
    # -------------------------------------#
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    # -------------------------------------#
    #   返回的inter的shape为[A,B]
    #   代表每一个真实框和先验框的交矩形
    # -------------------------------------#
    inter = intersect(box_a, box_b)
    # -------------------------------------#
    #   计算先验框和真实框各自的面积
    # -------------------------------------#
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    union = area_a + area_b - inter
    # -------------------------------------#
    #   每一个真实框和先验框的交并比[A,B]
    # -------------------------------------#
    return inter / union


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    # ----------------------------------------------#
    #   计算所有的先验框和真实框的重合程度
    # ----------------------------------------------#
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # ----------------------------------------------#
    #   所有真实框和先验框的最好重合程度
    #   best_prior_overlap [truth_box,1]
    #   best_prior_idx [truth_box,0]
    # ----------------------------------------------#
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    # ----------------------------------------------#
    #   所有先验框和真实框的最好重合程度
    #   best_truth_overlap [1,prior]
    #   best_truth_idx [1,prior]
    # ----------------------------------------------#
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    # ----------------------------------------------#
    #   用于保证每个真实框都至少有对应的一个先验框
    # ----------------------------------------------#
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    # ----------------------------------------------#
    #   获取每一个先验框对应的真实框[num_priors,4]
    # ----------------------------------------------#
    matches = truths[best_truth_idx]
    # Shape: [num_priors]
    conf = labels[best_truth_idx] + 1

    # ----------------------------------------------#
    #   如果重合程度小于threhold则认为是背景
    # ----------------------------------------------#
    conf[best_truth_overlap < threshold] = 0

    # ----------------------------------------------#
    #   利用真实框和先验框进行编码
    #   编码后的结果就是网络应该有的预测结果
    # ----------------------------------------------#
    loc = encode(matches, priors, variances)

    # [num_priors,4]
    loc_t[idx] = loc
    # [num_priors]
    conf_t[idx] = conf


def encode(matched, priors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])

    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


# -------------------------------------------------------------------#
#   Adapted from https://github.com/Hakuyume/chainer-ssd
#   利用预测结果对先验框进行调整，前两个参数用于调整中心的xy轴坐标
#   后两个参数用于调整先验框宽高
# -------------------------------------------------------------------#
def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


# -------------------------------------------------------------------#
#   Original author: Francisco Massa:
#   https://github.com/fmassa/object-detection.torch
#   Ported to PyTorch by Max deGroot (02/01/2017)
#   该部分用于进行非极大抑制，即筛选出一定区域内得分最大的框。
# -------------------------------------------------------------------#
def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union
        idx = idx[IoU.le(overlap)]
    return keep, count


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def ssd_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1)
    box_hw = np.concatenate((bottom - top, right - left), axis=-1)

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


MEANS = (104, 117, 123)


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, negatives_for_hard=100.0):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.negatives_for_hard = negatives_for_hard
        self.variance = Config['variance']

    def forward(self, predictions, targets):
        # --------------------------------------------------#
        #   取出预测结果的三个值：回归信息，置信度，先验框
        # --------------------------------------------------#
        loc_data, conf_data, priors = predictions
        # --------------------------------------------------#
        #   计算出batch_size和先验框的数量
        # --------------------------------------------------#
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        # --------------------------------------------------#
        #   创建一个tensor进行处理
        # --------------------------------------------------#
        loc_t = torch.zeros(num, num_priors, 4).type(torch.float32)
        conf_t = torch.zeros(num, num_priors).long()

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            priors = priors.cuda()

        for idx in range(num):
            # 获得真实框与标签
            truths = targets[idx][:, :-1]  # (n_obj,4)
            labels = targets[idx][:, -1]  # (n_obj)

            if (len(truths) == 0):
                continue

            # 获得先验框
            defaults = priors
            # --------------------------------------------------#
            #   利用真实框和先验框进行匹配。
            #   如果真实框和先验框的重合度较高，则认为匹配上了。
            #   该先验框用于负责检测出该真实框。
            # --------------------------------------------------#
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)

        # 所有conf_t>0的地方，代表内部包含物体
        pos = conf_t > 0

        # --------------------------------------------------#
        #   求和得到每一个图片内部有多少正样本
        #   num_pos  (num, )
        # --------------------------------------------------#
        num_pos = pos.sum(dim=1, keepdim=True)

        # --------------------------------------------------#
        #   取出所有的正样本，并计算loss
        #   pos_idx (num, num_priors, 4)
        # --------------------------------------------------#
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)  # 结束cxcywh 相关Loss
        # --------------------------------------------------#
        #   batch_conf  (num * num_priors, num_classes)
        #   loss_c      (num, num_priors)
        # --------------------------------------------------#
        batch_conf = conf_data.view(-1, self.num_classes)
        # 这个地方是在寻找难分类的先验框
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = loss_c.view(num, -1)

        # 难分类的先验框不把正样本考虑进去，只考虑难分类的负样本
        loss_c[pos] = 0
        # --------------------------------------------------#
        #   loss_idx    (num, num_priors)
        #   idx_rank    (num, num_priors)
        # --------------------------------------------------#
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # --------------------------------------------------#
        #   求和得到每一个图片内部有多少正样本
        #   num_pos     (num, )
        #   neg         (num, num_priors)
        # --------------------------------------------------#
        num_pos = pos.long().sum(1, keepdim=True)
        # 限制负样本数量
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        num_neg[num_neg.eq(0)] = self.negatives_for_hard
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # --------------------------------------------------#
        #   求和得到每一个图片内部有多少正样本
        #   pos_idx   (num, num_priors, num_classes)
        #   neg_idx   (num, num_priors, num_classes)
        # --------------------------------------------------#
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        # 选取出用于训练的正样本与负样本，计算loss
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        N = torch.max(num_pos.data.sum(), torch.ones_like(num_pos.data.sum()))
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c