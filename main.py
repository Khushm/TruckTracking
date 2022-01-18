import time
from loguru import logger
import cv2
import argparse
import numpy as np
from data_fetch.get_data import get_data
from utils.camera.camera import Camera
import json

channel = {}
camera_class = Camera(from_date_time=None, to_date_time=None, camera_no={}, panel_no=None)


def run():
    try:
        camera_class.infer()
        logger.info("Completed!")
    except Exception as e:
        logger.error("Error in run | {}".format(e))


if __name__ == "__main__":
    try:
        logger.info("Starting Up Application...")
        run()
    except Exception as e:
        logger.error("Error in starting application | {}".format(e))

# sudo docker run -dt --gpus all --env-file ./.env --name truck_tracker truck_tracking:0.3
# sudo docker build -t truck_tracking:0.3 .
