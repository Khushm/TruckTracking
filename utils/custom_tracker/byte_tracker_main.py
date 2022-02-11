import cv2
import argparse
from utils.custom_tracker.yolo import *
from loguru import logger
from utils.custom_tracker.timer import Timer
from utils.custom_tracker.custom_tracker.byte_tracker import BYTETracker
# from utils.custom_tracker.custom_tracker.mot_tracker import OnlineTracker
import numpy as np


class Tracker:
    def __init__(self):
        try:
            self.execution_path = os.getcwd()
            self.weights_path = os.path.join(self.execution_path, 'models', 'yolov3.weights')
            self.cfg_path = os.path.join(self.execution_path, 'models', 'yolov3.cfg')
            self.labels_path = os.path.join(self.execution_path, 'models', 'coco-labels')
            self.yoloh5_path = os.path.join(self.execution_path, 'models', 'yolo.h5')

            # self.threshold = 0.2
            # self.confidence = 0.5
            self.min_box_area = 10

            self.sub_dets = []
            self.frame_rate = 25
            self.fps = 1.
            self.timer = Timer()
            self.online_targets = []
            self.results = []
            self.tracker = BYTETracker(self.frame_rate)
        except Exception as e:
            logger.error("Error in initialising Tracker | {}".format(e))

    def infer(self, cam, sub_dets, frame, frame_counter):
        try:
            self.sub_dets = sub_dets
            if len(self.sub_dets) == 0:
                return
            self.online_targets = []
            self.results = []
            self.dets = np.array(self.sub_dets)
            self.frame = frame
            self.frame_counter = frame_counter
            self.height, self.width, _ = self.frame.shape
            self.img_size = [self.height, self.width]
            self.info_imgs = (self.height, self.width)

            for i in range(10):
                self.online_targets = self.tracker.update(self.dets, self.info_imgs, self.img_size)
                if len(self.online_targets) == len(self.sub_dets):
                    break
            # print("Online Targets: ", self.online_targets)
            self.online_tlwhs = []
            self.online_ids = []
            self.online_scores = []
            for t in self.online_targets:
                self.tlwh = t.tlwh
                self.tid = t.track_id
                if self.tlwh[2] * self.tlwh[3] > self.min_box_area:
                    self.online_tlwhs.append(self.tlwh)
                    self.online_ids.append(self.tid)
                    self.online_scores.append(t.score)
            self.results.append((self.frame_counter, self.online_tlwhs, self.online_ids, self.online_scores))
            self.timer.toc()
            self.frame_id = self.frame_counter
            self._id, self.frame1 = plot_tracking(cam, self.frame, self.online_tlwhs, self.online_ids, self.frame_id,
                                                 self.fps / self.timer.average_time)

            # print("ID: ", self._id)

            return self.results, self.frame
        except Exception as e:
            logger.error("Error in tracker infer | {}".format(e))
