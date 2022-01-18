from loguru import logger
from db.mongoConn import get_mongo_client
from datetime import datetime
Truckdata = {}


def update_data(online_ids, startX, startY, endX, endY, confidence, camera_no, panel_no, frame):
    try:
        Truckdata = {
            "truck_id": online_ids,
            "bounding_area": (startX, startY, endX, endY),
            "confidence": confidence,
            "camera_no": camera_no,
            "panel_no": panel_no,
            "frame": frame
        }
        # link['datetime_local'] = datetime.strptime(link['datetime_local'].split(".")[0], '%Y-%m-%dT%H:%M:%S')

        post_id = mongo_coll.insert_one(Truckdata)
        # post_id = mongo_coll.insert_one(Truckdata).inserted_id
        # logger.info("Data Inserted!")
    except Exception as e:
        logger.error("Error in dumping data | {}".format(e))


def ConnectMongo():
    while True:
        mongo_coll_ret = get_mongo_client()
        meta_data = get_meta_data()
        if mongo_coll_ret:
            logger.info("Connected to MongoDB successfully.")
            break
    meta_data_dict_ret = meta_data.find()
    return meta_data_dict_ret, mongo_coll_ret
