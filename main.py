import time
from loguru import logger
import argparse
import numpy as np
import json
from datetime import datetime
from db.mongoConn import get_mongo_client, load_meta_data
from os import getenv
from data_fetch.get_data import process_data
from db.mongoConn import MongoDB

coll = getenv("MONGO_COLLECTION_PRIMARY")  # db to load infer images
coll_sec = getenv("MONGO_COLLECTION_SEC")  # db to push results
meta_coll = getenv("MONGO_COLLECTION_META")  # db to load metadata

# load and process cam, panel from meta db & to-from date from environment variable
def run():
    try:
        from_dt = getenv("FROM_DT")
        to_dt = getenv("TO_DT")
        db = get_mongo_client()

        from_dt = datetime.strptime(from_dt, "%Y-%m-%dT%H:%M:%S.364Z")
        to_dt = datetime.strptime(to_dt, "%Y-%m-%dT%H:%M:%S.364Z")
        meta_col = load_meta_data(db[meta_coll])
        # meta_col = db_object.load_meta_data()

        for i, data in enumerate(meta_col):
            start = time.time()
            cam = data['channel_no']
            panel = data['panel_no']
            process_data(db[coll], db[coll_sec], from_dt, to_dt, cam, panel, i)
            # process_data(db_object, from_dt, to_dt, cam, panel, i)
            logger.info("Total time taken to infer image: {} CAM: {} Panel: {}".format(time.time() - start, cam, panel))
        logger.info("Completed!")
        get_mongo_client(dbClose=True)
        # db_object.close_connection()
    # close connection
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

# 21 Jan'22
# docker container logs tracker
# docker rm tracker
# docker run -dt --env-file ./.env --name tracker tracker:v18
# docker build -t tracker:v18 .
# docker image rm tracker:v16
# docker ps
# docker image ls
# docker-compose ps
# docker images

