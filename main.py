import time

from loguru import logger
import cv2
import argparse
import numpy as np
from data_fetch.get_data import get_data
from utils.camera.camera import Camera
import json

# camera = {5, 6, 7}
channel = {}


def run():
    try:
        load_json_file = open('read.json')
        json_data = json.load(load_json_file)
        for i, data in enumerate(json_data['TruckTrack']):
            start = time.time()
            channel[i] = Camera(data, i)
            channel[i].infer()
            logger.info("Total time taken to infer image: {} for camera: {}".format(time.time()-start, data['cam_no']))
            # camera[i].run(len(rows), thread_count, i)
        logger.info("Completed!")
    except Exception as e:
        logger.error("Error in run | {}".format(e))


if __name__ == "__main__":
    try:
        logger.info("Starting Up Application...")
        run()
    except Exception as e:
        logger.error("Error in starting application | {}".format(e))
