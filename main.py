import time
from loguru import logger
import cv2
import argparse
import numpy as np
import json
from datetime import datetime
from db.mongoConn import get_mongo_client, load_meta_data
from os import getenv
from data_fetch.get_data import process_data
coll = getenv("MONGO_COLLECTION_PRIMARY")
coll_sec = getenv("MONGO_COLLECTION_SEC")
meta_coll = getenv("MONGO_COLLECTION_META")


def run():
    try:
        db = get_mongo_client()
        # load_json_file = open('read.json')
        meta_col = load_meta_data(db[meta_coll])
        # json_data = json.load(load_json_file)
        for i, data in enumerate(meta_col):
            start = time.time()
            from_dt = datetime.strptime("2021-10-01T00:01:00.364Z", "%Y-%m-%dT%H:%M:%S.364Z")
            to_dt = datetime.strptime("2021-10-01T11:59:00.364Z", "%Y-%m-%dT%H:%M:%S.364Z")
            cam = data['channel_no']
            panel = data['panel_no']
            process_data(db[coll], db[coll_sec], from_dt, to_dt, cam, panel, i)
            logger.info(
                "Total time taken to infer image: {}".format(time.time() - start))
        logger.info("Completed!")

    #     close connection
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
