from db.mongoConn import get_mongo_client

Truckdata = {}


def update_data(link, online_ids):
    print(online_ids)
    # Truckdata = {
    #     "tid": _id,
    #     "bounding_area": (startX, startY, endX, endY),
    # }

    Truckdata.update(link)
    logger.info(Truckdata)
    post_id = mongo_coll.insert_one(Truckdata).inserted_id
    logger.info(post_id)


while True:
    mongo_coll = get_mongo_client()
    if mongo_coll:
        logger.info("Connected to MongoDB successfully.")
        break
