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


def argsparser():
    parser = argparse.ArgumentParser(description='Set some constants')
    parser.add_argument("--panel_no", type=int, default=500003, help="Set panel number")
    parser.add_argument("--channel_no", type=int, default=5, help="Set channel number")
    parser.add_argument(
        "--time",
        type=datetime,
        default=datetime.now() - timedelta(4),
        help="Get current datetime, to process previous day images"
    )
    parser.add_argument(
        "--api_request",
        default="https://api.smart-iam.com/api/image-store/metadata-v2",
        help="Link to post API, request to get query response"
    )
    parser.add_argument(
        "--start_time",
        type=str,
        default="00:01:00.364Z",
        help="choose the time, to start the process"
    )
    parser.add_argument(
        "--end_time",
        type=str,
        default="11:59:00.364Z",
        help="choose the time, to end the process"
    )
    return parser


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


# converting datetime object to given sting format
def time_string(date, time):
    return date.strftime("%Y-%m-%dT") + time


# print parsed arguments
def print_arguments(args):
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')


# List of all the frames between start and end time
final_data = []
valid_link = []


# post api request and return the list of all valid frames
def get_data(ch: int = 0):
    try:
        parser = argsparser()
        args = parser.parse_args()
        # processing on previous day frames
        print_arguments(args=args)
        query = {
            "time_intervals": [
                {
                    "start": time_string(args.time - timedelta(1), time=args.start_time),
                    "end": time_string(args.time - timedelta(1), time=args.end_time)
                }
            ],
            "panel_no": args.panel_no
        }
        response = requests.post(args.api_request, json=query)
        data = response.json()['data']

        args.channel_no = ch

        # filtering input for one particular channel
        for item in data:
            if item['channel_no'] == args.channel_no:
                final_data.append(url_image_converter(item))
                valid_link.append(item)
        logger.info('Length of valid images - {}'.format(len(final_data)))
        return final_data, valid_link
    except Exception as e:
        logger.error('ERROR | {}'.format(e))
