import cv2
import torch
import numpy as np

'''
For DataProcessing，Predicate
'''
def decode_letterbox(output_bboxes: np.ndarray, dx, dy, iw, ih, nw, nh):
    '''
    将output_bboxes 解析成(ih, iw, 3)下的坐标
    output_bboxes 必须是[[xm, ym, xa, ya],..]形式
    '''
    output_bboxes = output_bboxes.astype(np.float32)
    output_bboxes[:, [0, 2]] = (output_bboxes[:, [0, 2]] - dx) / nw * iw
    output_bboxes[:, [1, 3]] = (output_bboxes[:, [1, 3]] - dy) / nh * ih

    output_bboxes = np.round(output_bboxes).astype(np.int32)

    return output_bboxes

def normalize_image(img: np.ndarray):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(img) / 255.) - mean) / std

def valid_bboxes(bboxes: np.ndarray, ih, iw):
    '''
    只是规整化bbox
    bboxes: [[xm,ym,xa,ya],...] np.float32 (n,4)  ---->(n, 4)
    '''
    #将xm ym 小于0的置为0
    bboxes = bboxes.copy()
    bboxes[:, 0][bboxes[:, 0] < 0] = 0
    bboxes[:, 1][bboxes[:, 1] < 0] = 0
    #将xa超过iw的置为iw
    bboxes[:, 2][bboxes[:, 2] > iw] = iw
    bboxes[:, 3][bboxes[:, 3] > ih] = ih

    return bboxes

def select_valid_bboxes(bboxes: np.ndarray, ih, iw):
    '''
    bboxes: [[xm,ym,xa,ya],...] np.float32 (n,4)  ---->(*, 4)
    '''
    bboxes = valid_bboxes(bboxes, ih, iw)
    bboxes_w = bboxes[:, 2] - bboxes[:, 0]
    bboxes_h = bboxes[:, 3] - bboxes[:, 1]
    bboxes = bboxes[np.logical_and(bboxes_w > 1, bboxes_h > 1)]  # (*, 4)
    return bboxes

'''
For testing
'''
def show_pic(img, bboxes=None, name='pic'):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    img = img.copy()
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3)
    cv2.namedWindow(name, 0)  # 1表示原图
    cv2.moveWindow(name, 0, 0)
    # cv2.resizeWindow('pic', 1200,800)  # 可视化的图片大小
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
还需要修改！！！！！！！！！！！！！！！改save_checkpoint(...)
'''
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pth')     # 这里会存储迄今最优模型的参数
        # torch.save(model, 'finish_model.pkl')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss

if __name__ == '__main__':
    bboxes = np.array([[-1,1,2,2],[100, 100, 200, 200]])
    valid_bboxes1 = select_valid_bboxes(bboxes, 150, 150)
    print(valid_bboxes1)