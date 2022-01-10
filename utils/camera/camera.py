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
    def __init__(self, json_data, i):
        try:
            self.frame_counter = 0
            self.skip_frame = 1
            self.frame_list = []
            self.meta_data = []
            self.frame_link = []
            self.image_data = []
            self.uuid_mapping = dict()
            self.execution_path = os.getcwd()

            self.data = json_data
            self.i = i
            self.camera_no = self.data['cam_no']
            self.from_date_time = self.data['from_time']
            self.to_date_time = self.data['to_time']
            self.panel_no = self.data['panel_no']
            self.image_data, self.frame_link = get_data(self.camera_no, self.from_date_time, self.to_date_time,
                                                        self.panel_no)

            if not os.path.exists(os.path.join(self.execution_path, 'input_dir')):
                os.mkdir(os.path.join(self.execution_path, 'input_dir'))

            if not os.path.exists(os.path.join(self.execution_path, 'input_dir',
                                               str(self.camera_no) + '_' + str(self.i))):
                os.mkdir(os.path.join(self.execution_path, 'input_dir',
                                      str(self.camera_no) + '_' + str(self.i)))

            for frame, link in zip(self.image_data, self.frame_link):
                self.frame_counter += 1

                if frame is not None:
                    if self.frame_counter % self.skip_frame == 0:
                        frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
                        cv2.imwrite(os.path.join(self.execution_path, 'input_dir',
                                                 str(self.camera_no) + '_' + str(self.i),
                                                 str(self.frame_counter) + '.jpg'), frame)
                        self.frame_list.append(frame)
                        self.meta_data.append(link)
                    else:
                        continue
            self.tracker = Tracker()
            self.detector = PP_Detector()
            logger.info("Initialised Tracker & Detector../")
        except Exception as e:
            logger.error("Error in initialising Tracker & Detector! {}".format(e))

    def infer(self):
        try:
            self.frame_counter = 0
            for frame, link in zip(self.frame_list, self.meta_data):
                self.frame_counter += 1
                self.infer_img_path = os.path.join(self.execution_path, 'input_dir',
                                                   str(self.camera_no) + '_' + str(self.i),
                                                   str(self.frame_counter) + '.jpg')
                self.subdets, frame = self.detector.pp_infer_image_path(self.infer_img_path, frame)
                self.online_ids = self.tracker.infer(self.subdets, frame, self.frame_counter)
                # print('_________________')
                # logger.info(self.online_ids)
                if len(self.subdets) > 0:
                    for i in range(len(self.subdets)):
                        self.startX, self.startY, self.endX, self.endY, self.confidence = self.subdets[i]
                        if self.online_ids[i] not in self.uuid_mapping.keys():
                            self._id = str(uuid.uuid4())
                            self.uuid_mapping[self.online_ids[i]] = self._id
                        update_data(link, self.uuid_mapping[self.online_ids[i]],
                                    self.startX, self.startY,
                                    self.endX, self.endY, self.confidence)
            try:
                shutil.rmtree(os.path.join(self.execution_path, 'input_dir',
                                           str(self.camera_no) + '_' + str(self.i)))
                shutil.rmtree(os.path.join(self.execution_path, 'output'))
            except:
                pass
        except Exception as e:
            logger.error("Error in Camera Infer | {}".format(e))
