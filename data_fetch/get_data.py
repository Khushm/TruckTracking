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
# bbox_thresh = 10.2
intersection_thresh = 80
optical_thresh = 50

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


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
        return None

    except Exception as e:
        logger.debug('Error in Converting Url to Image:{}'.format(e))


def makeBoxPoints(x0, y0, x1, y1):
    old_pts = []
    y = y0
    while x0 < x1:
        x0 += 20
        while y0 < y1:
            y0 += 30
            old_pts.append([x0, y0])
        y0 = y
    old_pts = np.array([old_pts], dtype="float32")
    return old_pts


feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


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


def opticalFlow(frame_prev, frame_curr, old_pts):
    new_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    # old_pts = makeBoxPoints(old_pts[0], old_pts[1], old_pts[2], old_pts[3])
    old_pts = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_pts, None, **lk_params)
    dist = 0
    for i in range(len(new_pts[0])):
        # cv2.circle(mask, (int(new_pts[i][0][0]), int(new_pts[i][0][1])), 2, (0, 255, 0), 2)
        # combined = cv2.addWeighted(frame2, 0.7, mask, 0.3, 0.1)
        dist += pow(pow(new_pts[0][i][0] - old_pts[0][i][0], 2) + pow(new_pts[0][i][1] - old_pts[0][i][1], 2), 0.5)
    # cv2.imwrite(os.path.join('output', _id_uuid_mapping[online_ids[_][2][0]]['_id'], (str(ix) + '.jpg')), c)

    if dist / len(new_pts[0]) > optical_thresh:
        return True
    return False


# day-night 83/ 43Cam07
# new, old

def min_bbox(a: list, b: list):
    dx0 = min(a[0], b[0])
    dx2 = max(a[2], b[2])
    dy1 = min(a[1], b[1])
    dy2 = max(a[3], b[3])
    return [dx0, dy1, dx2, dy2]


def intersection_area(a: list, b: list):
    curr_image_area = ((abs(a[0] - a[2]) * abs(a[1] - a[3]) * 100) / frame_size)
    dx = abs(max(a[0], b[0]) - min(a[2], b[2]))
    dy = abs(max(a[1], b[1]) - min(a[3], b[3]))
    if (dx >= 0) and (dy >= 0):
        intersection_image_area = ((dx * dy * 100) / frame_size) / curr_image_area * 100
        # return intersection_image_area

        if intersection_image_area < 80:
            # same truck at same position or same truck just shifted
            return True
        else:
            # can be different truck at that position
            return False
    return None


# load images
# apply tracker if truck is detected else skip
# map id with uuid
def process_data(camera, panel, _id):
    try:
        # if camera == 7 or camera == 6:
        #     return
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
            read_curr_image = url_image_converter(data)
            if read_curr_image is None:
                print("Image not loaded!")
                break
            online_ids, f = tracker[_id].infer(camera, main_dets, read_curr_image, ix)
            # print(data['get_presignedUrl'], online_ids)
            if online_ids is None:
                continue
            for _ in range(len(online_ids)):
                x1, y1, w, h = [int(x) for x in online_ids[_][1][0]]
                x2 = x1 + w
                y2 = y1 + h
                _per = round((int(w * h) / frame_size) * 100, 2)
                # area
                print("Area-Thresh | Not Consider | ", _per)
                if _per < 3.45:
                    print(data['get_presignedUrl'])
                    continue
                if online_ids[_][2][0] not in _id_uuid_mapping.keys():
                    generate_uuid = str(uuid.uuid4())
                    _id_uuid_mapping[online_ids[_][2][0]] = {"_id": generate_uuid}

                # if camera == 7 and ix > 10:
                #     cv2.imshow("prev", url_image_converter(_id_uuid_mapping[online_ids[_][2][0]]))
                #     cv2.imshow("curr", url_image_converter(data))
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                flag = datetimeValidity(data["datetime_local"])
                if 'bbox' in _id_uuid_mapping[online_ids[_][2][0]]:
                    min_bbox_pts = min_bbox([x1, y1, x2, y2], _id_uuid_mapping[online_ids[_][2][0]]["bbox"])
                    min_x0, min_y0, min_x1, min_y1 = min_bbox_pts

                if 'area' in _id_uuid_mapping[online_ids[_][2][0]] and abs(
                        _id_uuid_mapping[online_ids[_][2][0]]["area"] - _per) > bbox_thresh:
                    print(_id_uuid_mapping[online_ids[_][2][0]]["area"], " :: ", _per)
                    print("Area Flag")

                    generate_uuid = str(uuid.uuid4())
                    _id_uuid_mapping[online_ids[_][2][0]] = {"_id": generate_uuid}
                elif 'get_presignedUrl' in _id_uuid_mapping[online_ids[_][2][0]] and flag and similar_images(
                        url_image_converter(_id_uuid_mapping[online_ids[_][2][0]])[min_y0:min_y1, min_x0:min_x1],
                        read_curr_image[min_y0:min_y1, min_x0:min_x1], camera, ix) < 4.9:
                    print("Similarity Flag")
                    generate_uuid = str(uuid.uuid4())
                    _id_uuid_mapping[online_ids[_][2][0]] = {"_id": generate_uuid}
                # intersection_area_diff = intersection_area([x1, y1, x2, y2], _id_uuid_mapping[online_ids[_][2][0]]["bbox"])
                # elif 'bbox' in _id_uuid_mapping[online_ids[_][2][0]] and intersection_area(
                #         [x1, y1, x2, y2], _id_uuid_mapping[online_ids[_][2][0]]["bbox"]):
                #     pass
                # elif 'bbox' in _id_uuid_mapping[online_ids[_][2][0]] and opticalFlow(url_image_converter(_id_uuid_mapping[online_ids[_][2][0]]),
                #                    url_image_converter(data), _id_uuid_mapping[online_ids[_][2][0]]["bbox"]):
                #     generate_uuid = str(uuid.uuid4())
                #     _id_uuid_mapping[online_ids[_][2][0]] = {"_id": generate_uuid}

                object_list = [{'name': 'truck', 'box_points': [x1, y1, x2, y2]}]
                _id_uuid_mapping[online_ids[_][2][0]].update({"get_presignedUrl": data['get_presignedUrl']})
                _id_uuid_mapping[online_ids[_][2][0]].update({"area": round((int(w * h) / frame_size) * 100, 2)})
                _id_uuid_mapping[online_ids[_][2][0]].update({"bbox": [x1, y1, x2, y2]})

                print(_id_uuid_mapping[online_ids[_][2][0]]['_id'])
                print(data["datetime_local"])
                print(data['get_presignedUrl'])
                print(camera)
                print("__________________________________")

                push_data(_id_uuid_mapping[online_ids[_][2][0]]['_id'], data, object_list)
    except Exception as e:
        logger.error('Error in fetching data | {}'.format(e))
# 28 25 24 23 22
# 21

# 28 25 23 22 thresh>[
# image not loaded 21
# thresh 27
# 24

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
