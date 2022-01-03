from loguru import logger
import cv2
import argparse
import os
import numpy as np
from data_fetch.get_data import get_data
from utils.object_detection.pp_yolo_main import PPYOLO
from utils.custom_tracker.byte_tracker_main import Tracker
from db.push_data import update_data
import uuid


class Camera:
    def __init__(self, ch):
        try:
            self.frame_counter = 0
            self.skip_frame = 1
            self.frame_list = []
            self.meta_data = []
            self.frame_link = []
            self.image_data = []
            self.uuid_mapping = dict()
            self.channel_no = ch
            self.image_data, self.frame_link = get_data(ch)
            self.execution_path = os.getcwd()
            for frame, link in zip(self.image_data, self.frame_link):
                self.frame_counter += 1

                if frame is not None:
                    if self.frame_counter % self.skip_frame == 0:
                        frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
                        cv2.imwrite(os.path.join(self.execution_path, 'input_dir', str(self.frame_counter) + '.jpg'),
                                    frame)
                        self.frame_list.append(frame)
                        self.meta_data.append(link)
                    else:
                        continue
                else:
                    logger.info('Reconnecting to Camera.../')
                    break
            self.tracker = Tracker()
            self.detector = PPYOLO()
        except Exception as e:
            logger.error(e)

    def infer(self):
        try:
            self.frame_counter = 0
            for frame, link in zip(self.frame_list, self.meta_data):
                self.frame_counter += 1
                self.infer_img_path = os.path.join(self.execution_path, 'input_dir', str(self.frame_counter) + '.jpg')
                self.subdets, frame = self.detector.infer(self.infer_img_path, frame)
                self.online_ids = self.tracker.infer(self.subdets, frame, self.frame_counter)
                # print('_________________')
                logger.info(self.online_ids)
                logger.info(self.subdets)
                if len(self.subdets) > 0:
                    for i in range(len(self.subdets)):
                        self.startX, self.startY, self.endX, self.endY, self.confidence = self.subdets[i]
                        if self.online_ids[0] in self.uuid_mapping.keys():
                            update_data(link, self.uuid_mapping[self.online_ids[i]], self.startX, self.startY,
                                        self.endX, self.endY,
                                        self.confidence)
                        else:
                            self._id = str(uuid.uuid4())
                            self.uuid_mapping[self.online_ids[i]] = self._id
                            update_data(link, self.uuid_mapping[self.online_ids[i]], self.startX, self.startY,
                                        self.endX, self.endY, self.confidence)
        except Exception as e:
            logger.error(e)
