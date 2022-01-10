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

ssl._create_default_https_context = ssl._create_unverified_context

api_request_link = "https://api.smart-iam.com/api/image-store/metadata-v2"

# List of all the frames between start and end time
final_data = []
valid_link = []


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


# post api request and return the list of all valid frames
def get_data(camera_no, from_time, to_time, panel_no, ai_id=1):
    try:
        final_data = []
        # processing on previous day frames
        query = {
            "time_intervals": [
                {
                    "start": from_time,
                    "end": to_time
                }
            ],
            "panel_no": panel_no
        }
        response = requests.post(api_request_link, json=query)
        data = response.json()['data']

        # filtering input for one particular channel
        for item in data:
            if item['channel_no'] == camera_no:
            # if item['channel_no'] == camera_no and item['ai_id'] == ai_id:
                final_data.append(url_image_converter(item))
                valid_link.append(item)
        logger.info('Length of valid images - {}'.format(len(final_data)))
        return final_data, valid_link
    except Exception as e:
        logger.error('Error in fetching data | {}'.format(e))
