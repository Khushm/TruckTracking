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

ssl._create_default_https_context = ssl._create_unverified_context
api_request_link = "https://api.smart-iam.com/api/image-store/metadata-v2"
frame_size = 1080*1920
# List of all the frames between start and end time
box_points = []
url_list = []
final_data = []
probability = []
meta_data = []
tracker = {}
truck_thresh = 10
# truck_thresh = 200000


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


def is_visual_appearance_not_same(current_, previous_, i):
    current = cv2.cvtColor(current_, cv2.COLOR_BGR2GRAY)
    previous = cv2.cvtColor(previous_, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(current, previous)
    diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    h, w = diff.shape
    tot_pix_count = h * w

    # cv2.imshow("diff", diff)
    # cv2.imshow("C", current_)
    # cv2.imshow("P", previous_)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    Count = {"black_pixel_count": float(np.sum(diff == 0) / tot_pix_count) * 100,
             "white_pixel_count": float(np.sum(diff == 255) / tot_pix_count) * 100}
    if Count["white_pixel_count"] > 22:
        return True
    return False


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
            online_ids, f = tracker[_id].infer(camera, main_dets, url_image_converter(data), ix)
            # print(data['get_presignedUrl'], online_ids)
            if online_ids is None:
                continue
            for _ in range(len(online_ids)):
                x1, y1, w, h = [int(x) for x in online_ids[_][1][0]]
                x2 = x1 + w
                y2 = y1 + h
                if online_ids[_][2][0] not in _id_uuid_mapping.keys():
                    generate_uuid = str(uuid.uuid4())
                    _id_uuid_mapping[online_ids[_][2][0]] = {"_id": generate_uuid}

                _per = round((int(w * h)/frame_size)*100, 2)
                if 'area' in _id_uuid_mapping[online_ids[_][2][0]] and abs(
                        _id_uuid_mapping[online_ids[_][2][0]]["area"] - _per) > truck_thresh:
                    generate_uuid = str(uuid.uuid4())
                    _id_uuid_mapping[online_ids[_][2][0]] = {"_id": generate_uuid}

                object_list = [{'name': 'truck', 'box_points': [x1, y1, x2, y2]}]
                _id_uuid_mapping[online_ids[_][2][0]].update({"get_presignedUrl": data['get_presignedUrl']})
                _id_uuid_mapping[online_ids[_][2][0]].update({"area": round((int(w * h)/frame_size)*100, 2)})

                # if not os.path.exists(os.path.join("output", _id_uuid_mapping[online_ids[_][2][0]]['_id'])):
                #     os.mkdir(os.path.join("output", _id_uuid_mapping[online_ids[_][2][0]]['_id']))
                # c = url_image_converter(data)
                # c = cv2.rectangle(c, [x1, y1], [x2, y2], color=(255, 0, 0), thickness=2)

                # if ix > 15:
                #     cv2.imshow("id", url_image_converter(_id_uuid_mapping[online_ids[_]]))
                #     cv2.imshow("d", c)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()

                # cv2.imwrite(os.path.join('output', _id_uuid_mapping[online_ids[_][2][0]]['_id'], (str(ix) + '.jpg')), c)
                push_data(_id_uuid_mapping[online_ids[_][2][0]]['_id'], data, object_list)
    except Exception as e:
        logger.error('Error in fetching data | {}'.format(e))
# 28 25 24 23 22
# 21
