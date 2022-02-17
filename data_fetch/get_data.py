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
from utils.custom_tracker.byte_tracker_main import Tracker
from how_similar_two_images_are import similar_images

ssl._create_default_https_context = ssl._create_unverified_context
api_request_link = "https://api.smart-iam.com/api/image-store/metadata-v2"
frame_size = 1080 * 1920
# List of all the frames between start and end time
box_points = []
url_list = []
final_data = []
probability = []
meta_data = []
tracker = {}

bbox_thresh = 7.4
intersection_thresh = 80
optical_thresh = 50


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
        return None

    except Exception as e:
        logger.debug('Error in Converting Url to Image:{}'.format(e))


def datetimeValidity(d):
    # check morning & evening condition
    am6 = 6
    am8 = 8
    pm5 = 18
    pm7 = 19
    if am6 <= d.hour <= am8:
        return 0
    if pm5 <= d.hour <= pm7:
        return 0
    return 1


# new, old
def min_bbox(a: list, b: list):
    dx0 = min(a[0], b[0])
    dx2 = max(a[2], b[2])
    dy1 = min(a[1], b[1])
    dy2 = max(a[3], b[3])
    return [dx0, dy1, dx2, dy2]


# load images
# apply tracker if truck is detected else skip
# map id with uuid
def process_data(camera, panel, _id):
    try:
        tracker[_id] = Tracker()
        _id_uuid_mapping = dict()
        final_data = fetch_data(camera, panel)

        for (ix, data) in enumerate(final_data):
            main_dets = []
            try:
                object_set = set(data['object_list'])
            except:
                continue
            #     processing further only if truck is detected
            if 'truck' not in object_set:
                continue
            for i in data['objects']:
                if i['name'] == 'truck':
                    sub_dets = i['box_points'].copy()
                    if (sub_dets[2] - sub_dets[0]) * (sub_dets[3] - sub_dets[1]) < 20500:
                        continue
                    sub_dets.append(i['percentage_probability'] * 0.01)
                    main_dets.append(sub_dets)
            if len(main_dets) == 0:
                continue
            read_curr_image = url_image_converter(data)
            if read_curr_image is None:
                print("Image not loaded!")
                break
            online_ids, f = tracker[_id].infer(camera, main_dets, read_curr_image, ix)

            if online_ids is None:
                continue
            for _ in range(len(online_ids)):
                x1, y1, w, h = [int(x) for x in online_ids[_][1][0]]
                x2 = x1 + w
                y2 = y1 + h
                _per = round((int(w * h) / frame_size) * 100, 2)
                # if area of bbox is less than 3.45 continue
                if _per < 3.45:
                    continue
                # generate id if not present in dictionary
                if online_ids[_][2][0] not in _id_uuid_mapping.keys():
                    generate_uuid = str(uuid.uuid4())
                    _id_uuid_mapping[online_ids[_][2][0]] = {"_id": generate_uuid}

                # checking the date of particular image to avoid similarity check in morning and at evening
                flag = datetimeValidity(data["datetime_local"])

                # extract in max area box-points of truck from currect and previous frame
                if 'bbox' in _id_uuid_mapping[online_ids[_][2][0]]:
                    min_bbox_pts = min_bbox([x1, y1, x2, y2], _id_uuid_mapping[online_ids[_][2][0]]["bbox"])
                    min_x0, min_y0, min_x1, min_y1 = min_bbox_pts

                # if area is greater than threshold consider it as new truck and update new id
                # if check then go for similarity condition
                # if value is less than given threshold then it is a new truck
                # similarity is check only if the flag for time condition is set to true
                if 'area' in _id_uuid_mapping[online_ids[_][2][0]] and abs(
                        _id_uuid_mapping[online_ids[_][2][0]]["area"] - _per) > bbox_thresh:
                    generate_uuid = str(uuid.uuid4())
                    _id_uuid_mapping[online_ids[_][2][0]] = {"_id": generate_uuid}
                elif 'get_presignedUrl' in _id_uuid_mapping[online_ids[_][2][0]] and flag and similar_images(
                        url_image_converter(_id_uuid_mapping[online_ids[_][2][0]])[min_y0:min_y1, min_x0:min_x1],
                        read_curr_image[min_y0:min_y1, min_x0:min_x1]) < 4.9:
                    generate_uuid = str(uuid.uuid4())
                    _id_uuid_mapping[online_ids[_][2][0]] = {"_id": generate_uuid}

                object_list = [{'name': 'truck', 'box_points': [x1, y1, x2, y2]}]
                _id_uuid_mapping[online_ids[_][2][0]].update({"get_presignedUrl": data['get_presignedUrl']})
                _id_uuid_mapping[online_ids[_][2][0]].update({"area": round((int(w * h) / frame_size) * 100, 2)})
                _id_uuid_mapping[online_ids[_][2][0]].update({"bbox": [x1, y1, x2, y2]})
                push_data(_id_uuid_mapping[online_ids[_][2][0]]['_id'], data, object_list)
    except Exception as e:
        logger.error('Error in fetching data | {}'.format(e))