import time
import uuid
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
from db.mongoConn import fetch_data, push_data
# from db.mongoConnSec import get_mongo_client
from utils.custom_tracker.byte_tracker_main import Tracker

ssl._create_default_https_context = ssl._create_unverified_context

api_request_link = "https://api.smart-iam.com/api/image-store/metadata-v2"

# List of all the frames between start and end time
box_points = []
url_list = []
final_data = []
probability = []
meta_data = []
tracker = {}


def url_image_converter(data):
    try:
        for i in range(0, 2):
            image_url = data['get_presignedUrl']
            readFlag = cv2.IMREAD_COLOR

            try:
                resp = urlopen(image_url, timeout=7)
            except Exception as e:
                logger.info('Could not load image from URl in: 7 seconds: {}'.format(e))
                continue
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, readFlag)

            if image is None:
                logger.info('Image received from URL is None, wiating for 30 sec.../')
                time.sleep(30)
                continue
            else:
                return image
        return 'null'

    except Exception as e:
        logger.debug('Error in Converting Url to Image:{}'.format(e))


# load images
# apply tracker if truck is detected else skip
# map id with uuid
def process_data(coll, dump_col, from_time, to_time, camera, panel, _id):
# def process_data(db_object, from_time, to_time, camera, panel, _id):
    try:
        tracker[_id] = Tracker()
        _id_uuid_mapping = dict()
        final_data = fetch_data(coll, from_time, to_time, camera, panel)

        for (ix, data) in enumerate(final_data):
            main_dets = []
            try:
                object_set = set(data['object_list'])
            except:
                continue
            if 'truck' not in object_set:
                continue
            for i in data['objects']:
                if i['name'] == 'truck':
                    sub_dets = i['box_points'].copy()
                    sub_dets.append(i['percentage_probability']*0.01)
                    main_dets.append(sub_dets)

            online_ids = tracker[_id].infer(main_dets, url_image_converter(data), ix)
            # print(data['get_presignedUrl'], online_ids)
            for _ in range(len(online_ids)):
                if online_ids[_] not in _id_uuid_mapping.keys():
                    generate_uuid = str(uuid.uuid4())
                    _id_uuid_mapping[online_ids[_]] = generate_uuid
                push_data(dump_col, _id_uuid_mapping[online_ids[_]], data)

    except Exception as e:
        logger.error('Error in fetching data | {}'.format(e))
