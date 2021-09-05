import cv2
from nets.ssd import SSDDetectionFactory

if __name__ == '__main__':
    factory = SSDDetectionFactory()
    ssd_detection_net = factory.get_detection_net()

    img = cv2.imread('img/csgo.jpg')
    ssd_detection_net.test_step(img)

    ssd_trainer = factory.get_detection_trainer(annotation_path='2007_train.txt', max_epoch=10, batch_size=5)
    ssd_trainer.fit()