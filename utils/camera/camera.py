from loguru import logger
import cv2
import argparse
import os
import numpy as np
from data_fetch.get_data import get_data
from utils.object_detection.PP_Yolo_Detector.pp_detector import PP_Detector
from utils.custom_tracker.byte_tracker_main import Tracker
from db.push_data import update_data
import uuid
import shutil


class Camera:
    def __init__(self, from_date_time, to_date_time, camera_no, panel_no):
        try:
            self.frame_counter = 0
            self.skip_frame = 1
            self.frame_list = []
            self.meta_data = []
            self.frame_link = []
            self.image_data = []
            self.uuid_mapping = dict()
            self.execution_path = os.getcwd()

            self.data = {}
            self.i = 0
            self.camera_no: dict = camera_no
            self.from_date_time: datetime = from_date_time
            self.to_date_time: datetime = to_date_time
            self.panel_no: int = panel_no

            self.tracker = []
            self.counter = 0
            self.tracker = Tracker()
            logger.info("Initialised Tracker & Detector../")
        except Exception as e:
            logger.error("Error in initialising Tracker & Detector! {}".format(e))

    def infer(self):
        try:
            to_date_time = self.to_date_time.replace(day=self.from_date_time.day)
            while self.to_date_time >= to_date_time:
                for camera_no in self.camera_no:
                    self.counter += 1
                    self.tracker[counter] = Tracker()
                    self.uuid_mapping = dict()
                    self.image_data, self.box_points, self.probability = \
                        get_data(self.from_date_time, self.to_date_time, camera_no, self.panel_no)

                    self.from_date_time = self.from_date_time + timedelta(1)
                    to_date_time = self.to_date_time.replace(day=self.from_date_time.day)
                    self.frame_counter = 0
                    for frame, b_points, prob in zip(self.image_data, self.box_points, self.probability):
                        self.frame_counter += 1
                        if frame is not None:
                            if self.frame_counter % self.skip_frame == 0:
                                frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
                                self.subdets = b_points
                                self.subdets.append(prob)
                                self.online_ids = self.tracker[counter].infer(self.subdets, frame, self.frame_counter)
                            else:
                                continue
                                self.startX, self.startY, self.endX, self.endY, self.confidence = self.subdets
                                if self.online_ids not in self.uuid_mapping.keys():
                                    self._id = str(uuid.uuid4())
                                    self.uuid_mapping[self.online_ids] = self._id
                                update_data(self.uuid_mapping[self.online_ids], self.startX, self.startY,
                                            self.endX, self.endY, self.confidence, camera_no, self.panel_no, frame)
        except Exception as e:
            logger.error("Error in Camera Infer | {}".format(e))
