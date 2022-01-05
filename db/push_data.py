from loguru import logger
from db.mongoConn import get_mongo_client

Truckdata = {}


def update_data(link, online_ids, startX, startY, endX, endY, confidence):
    try:
        # logger.info(online_ids)
        Truckdata = {
            "truck_id": online_ids,
            "bounding_area": (startX, startY, endX, endY),
            "confidence": confidence
        }
        link['prev_mongo_id'] = link['_id']
        del link['_id']
        Truckdata.update(link)
        post_id = mongo_coll.insert_one(Truckdata)
        # post_id = mongo_coll.insert_one(Truckdata).inserted_id
        logger.info(online_ids, post_id)
    except Exception as e:
        logger.error(e)


while True:
    mongo_coll = get_mongo_client()
    if mongo_coll:
        logger.info("Connected to MongoDB successfully.")
        break
