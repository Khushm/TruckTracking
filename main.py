from loguru import logger
import cv2
import argparse
import numpy as np
from data_fetch.get_data import get_data
# from db.push_data import update_data
from utils.camera.camera import Camera

camera = {5, 6, 7}
channel = {}


def run():
    try:
        for i in camera:
            channel[i] = Camera(i)
            channel[i].infer()
            # camera[i].run(len(rows), thread_count, i)
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    try:
        logger.info("Starting Up Application...")
        run()
    except Exception as e:
        logger.error(e)
