import time
from datetime import datetime, timedelta
import json
import cv2
import os
from urllib.request import urlopen
import requests
from loguru import logger
import numpy as np
import ssl
import argparse
from db.push_data import ConnectMongo

ssl._create_default_https_context = ssl._create_unverified_context

api_request_link = "https://api.smart-iam.com/api/image-store/metadata-v2"

# List of all the frames between start and end time
box_points = []
url_list = []
final_data = []
probability = []


def get_data(from_time, to_time, camera, panel):
    try:
        # tracker = Tracker()
        final_data, mongo_col = ConnectMongo()
        for (i,data) in enumerate(final_data):
            object_set = set(data['object_list'])
            sub_dets = []
            main_det = []
            if (data['datetime_local'] >= from_time) and (data['datetime_local'] <= to_time) \
                    and data['panel_no'] == panel and data['channel_no'] == camera and 'truck' in object_set:
                for i in range(len(data['objects'])):
                    for key, value in i.items():
                        if value['name'] == 'truck':
                            url_list.append(data['get_presignedUrl'])
                            box_points.append(value['box_points'])
                            probability.append(value['percentage_probability'])
                            sub_dets = [value['box_points'][0],value['box_points'][1],value['box_points'][2],value['box_points'][3],value['percentage_probability']]
                            main_det.append(sub_dets)

            # Apply tracker
            # tracker.infer(main_det, frame, i)


        return url_list, box_points, probability
    except Exception as e:
        logger.error('Error in fetching data | {}'.format(e))
