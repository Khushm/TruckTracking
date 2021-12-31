from utils.object_detection.PaddleDetection.tools.infer import main
import cv2
import argparse
from loguru import logger
import numpy as np


class PPYOLO:
    def __init__(self):
        self.infer_img_path = os.path.join(execution_path, 'input_dir', str(frame_counter) + '.jpg')
        self.infer_weights = 'https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams'
        self.boxes = []
        self.classids = []
        self.confidence = []
        self.sub_dets = []

    def infer(self, img_path, frame):
        self.infer_img_path = img_path
        self.frame = frame
        self.boxes, self.classids, self.confidence = main(self.infer_img_path, self.infer_weights)

        if len(self.boxes) > 0:
            for i in range(len(self.boxes)):
                if self.classids[i] != 'truck':
                    continue

                self.x, self.y, self.w, self.h = self.boxes[i]

                self.startX = int(self.x)
                self.endX = int(self.x + self.w)
                self.startY = int(self.y)
                self.endY = int(self.y + self.h)

                self.cords_data = [self.startX, self.startY, self.endX, self.endY, self.confidence[i]]
                cv2.rectangle(self.frame, (self.startX, self.startY), (self.endX, self.endY), color=(0, 0, 255),
                              thickness=1)
                self.sub_dets.append(self.cords_data)
        return self.sub_dets, self.frame
