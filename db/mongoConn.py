from pymongo import MongoClient
from loguru import logger
from os import getenv
from urllib.parse import quote_plus
from datetime import datetime

user = getenv("MONGO_USERNAME_PRIMARY")
password = getenv("MONGO_PASSWORD_PRIMARY")
host = str(getenv("MONGO_HOST_PRIMARY"))
db = getenv("MONGO_DATABASE_PRIMARY")
# coll = getenv("MONGO_COLLECTION_PRIMARY")  # db to load infer images
coll_sec = getenv("MONGO_COLLECTION_SEC")  # db to push results
meta_coll = getenv("MONGO_COLLECTION_META")  # db to load metadata

user_p = getenv("MONGO_USERNAME_PROD")
password_p = getenv("MONGO_PASSWORD_PROD")
host_p = str(getenv("MONGO_HOST_PROD"))
db_p = getenv("MONGO_DATABASE_PROD")
coll_p = getenv("MONGO_COLLECTION_PROD")

MONGO_URL_PROD = "mongodb://%s:%s@%s" % (quote_plus(user_p), quote_plus(password_p), host_p)
MONGO_URL = "mongodb://%s:%s@%s" % (quote_plus(user), quote_plus(password), host)
mongo_client = None
infer_images_collection = None
metadata_collection = None
result_collection = None


# connect to db
def get_mongo_client():
    try:
        global mongo_client, metadata_collection, result_collection, infer_images_collection
        mongo_client = MongoClient(MONGO_URL)
        mongo_client.admin.authenticate(user, password)
        database = mongo_client[db]
        metadata_collection = database[meta_coll]
        result_collection = database[coll_sec]
        # infer_images_collection = database[coll]
    except Exception as e:
        logger.debug(f'Error while Connecting to Mongo Client: | Error:{e}')
        raise e


def get_mongo_client_prod():
    try:
        global mongo_client_prod, infer_images_collection
        mongo_client_prod = MongoClient(MONGO_URL_PROD)
        mongo_client_prod.admin.authenticate(user_p, password_p)
        database = mongo_client_prod[db_p]
        infer_images_collection = database[coll_p]
    except Exception as e:
        logger.debug(f'Error while Connecting to Mongo Client: | Error:{e}')
        raise e


def close_connection():
    global mongo_client
    if mongo_client is not None:
        mongo_client.close()
    else:
        pass


# fetch images between given from-to date_time
def fetch_data(camera, panel):
    try:
        global infer_images_collection
        meta_data_dict_ret = infer_images_collection.find({
            'datetime_local': {
                '$gte': datetime.strptime(getenv("FROM_DT"), "%Y-%m-%dT%H:%M:%S.364Z"),
                '$lte': datetime.strptime(getenv("TO_DT"), "%Y-%m-%dT%H:%M:%S.364Z"),
            },
            'panel_no': panel,
            'channel_no': camera
        })
        return meta_data_dict_ret
    except Exception as e:
        logger.error(e)


# load meta data of particular id
def load_meta_data(ai_id=4):
    try:
        global metadata_collection
        meta_data_dict_ret = metadata_collection.find({
            'ai_id': ai_id
        })
        return meta_data_dict_ret
    except Exception as e:
        logger.error(e)


# push results back to db
def push_data(online_ids, data, object_list):
    try:
        global result_collection
        try:
            data['prev_mongo_id'] = data['_id']
            del data['_id']
            data['datetime_local'] = datetime.strptime(data['datetime_local'], "%Y-%m-%dT%H:%M:%S.364Z")

        except:
            pass
        temp = {}
        temp['_uuid'] = online_ids
        temp['truck_list'] = object_list
        data.update(temp)
        # data.update(object_list)
        post_id = result_collection.insert_one(data)

        # post_id = mongo_coll.insert_one(Truckdata).inserted_id
        # logger.info("Data Inserted!")
    except Exception as e:
        logger.error("Error in dumping data | {}".format(e))
