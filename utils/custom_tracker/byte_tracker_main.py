import cv2
import argparse
from utils.custom_tracker.yolo import *
from loguru import logger
from utils.custom_tracker.timer import Timer
from utils.custom_tracker.custom_tracker.byte_tracker import BYTETracker
import numpy as np


class Tracker:
    def __init__(self):
        self.execution_path = os.getcwd()
        self.weights_path = os.path.join(self.execution_path, 'models', 'yolov3.weights')
        self.cfg_path = os.path.join(self.execution_path, 'models', 'yolov3.cfg')
        self.labels_path = os.path.join(self.execution_path, 'models', 'coco-labels')
        self.yoloh5_path = os.path.join(self.execution_path, 'models', 'yolo.h5')

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-m', '--model-path', type=str, default='./yolov2-coco/',
                                 help='base path to YOLO directory where models wieghts are.')
        self.parser.add_argument('-w', '--weights', type=str, default='./yolov3-coco/yolov3.weights',
                                 help='Path to the file which contains the weights.')
        self.parser.add_argument('-cfg', '--config', type=str, default='./yolov3-coco/yolov3.cfg',
                                 help='Path to the configuration file for the YOLOv3 model.')
        self.parser.add_argument('-vo', '--video-output-path', type=str, default='./output/output.avi',
                                 help='The path of the output video file')
        self.parser.add_argument('-l', '--labels', type=str, default='./yolov3-coco/coco-labels',
                                 help='Path to file having label')
        self.parser.add_argument('-c', '--confidence', type=float, default=0.5,
                                 help='probability limit')
        self.parser.add_argument('-th', '--threshold', type=float, default=0.2,
                                 help='Threshold to apply the non-max supression')
        self.parser.add_argument('--download-model', type=bool, default=False,
                                 help='Set to true, if the model weights and cfg are not present')
        self.parser.add_argument('-t', '--show-time', type=bool, default=False,
                                 help='Show the time taken to infer each image.')

        # Tracker Constants:
        self.parser.add_argument("--track_thresh", type=float, default=0.4,
                                 help="tracking confidence threshold")
        self.parser.add_argument("--track_buffer", type=int, default=10,
                                 help="the frames for keep lost tracks")
        self.parser.add_argument("--match_thresh", type=int, default=0.8,
                                 help="matching threshold for tracking")
        self.parser.add_argument('--min-box-area', type=float,
                                 default=10, help='filter out tiny boxes')
        self.parser.add_argument("--mot20", dest="mot20", default=False,
                                 action="store_true", help="test mot20.")
        self.sub_dets = []
        self.frame_rate = 25
        self.fps = 1.
        self.timer = Timer()
        self.online_targets = []
        self.results = []
        self.FLAGS, self.unparsed = self.parser.parse_known_args()
        self.tracker = BYTETracker(self.FLAGS, self.frame_rate)

    def infer(self, sub_dets, frame, frame_counter):
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
            print("Online Targets: ", self.online_targets)
            self.online_tlwhs = []
            self.online_ids = []
            self.online_scores = []
            for t in self.online_targets:
                self.tlwh = t.tlwh
                self.tid = t.track_id
                if self.tlwh[2] * self.tlwh[3] > self.FLAGS.min_box_area:
                    self.online_tlwhs.append(self.tlwh)
                    self.online_ids.append(self.tid)
                    self.online_scores.append(t.score)
            self.results.append((self.frame_counter, self.online_tlwhs, self.online_ids, self.online_scores))
            self.timer.toc()
            self.frame_id = self.frame_counter
            self._id, self.frame = plot_tracking(self.frame, self.online_tlwhs, self.online_ids, self.frame_id,
                                                 self.fps / self.timer.average_time)

            print("ID: ", self._id)
            return self.online_ids
        except Exception as e:
            logger.error(e)
