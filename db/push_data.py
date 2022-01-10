from loguru import logger
from db.mongoConn import get_mongo_client

Truckdata = {}


def update_data(link, online_ids, startX, startY, endX, endY, confidence):
    try:
        Truckdata = {
            "truck_id": online_ids,
            "bounding_area": (startX, startY, endX, endY),
            "confidence": confidence
        }
        try:
            link['prev_mongo_id'] = link['_id']
            del link['_id']
        except:
            pass
        Truckdata.update(link)
        post_id = mongo_coll.insert_one(Truckdata)
        # post_id = mongo_coll.insert_one(Truckdata).inserted_id
        logger.info(post_id)
    except Exception as e:
        logger.error("Error in dumping data | {}".format(e))


while True:
    mongo_coll = get_mongo_client()
    if mongo_coll:
        logger.info("Connected to MongoDB successfully.")
        break
