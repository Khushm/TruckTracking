import time
from loguru import logger
import argparse
import numpy as np
from db.mongoConn import get_mongo_client, load_meta_data, close_connection, get_mongo_client_prod
from data_fetch.get_data import process_data


# load and process cam, panel from meta db & to-from date from environment variable
def run():
    try:
        get_mongo_client()
        get_mongo_client_prod()
        meta_col = load_meta_data()

        for i, data in enumerate(meta_col):
            start = time.time()
            cam = data['channel_no']
            panel = data['panel_no']
            process_data(cam, panel, i)
            logger.info("Total time taken to infer image: {} CAM: {} Panel: {}".format(time.time() - start, cam, panel))
        logger.info("Completed!")
        close_connection()
    except Exception as e:
        logger.error("Error in run | {}".format(e))


if __name__ == "__main__":
    try:
        logger.info("Starting Up Application...")
        run()
    except Exception as e:
        logger.error("Error in starting application | {}".format(e))
