from pymongo import MongoClient
from loguru import logger
from os import getenv
from urllib.parse import quote_plus
from datetime import datetime, timedelta

user = getenv("MONGO_USERNAME_PRIMARY")
password = getenv("MONGO_PASSWORD_PRIMARY")
host = str(getenv("MONGO_HOST_PRIMARY"))
db = getenv("MONGO_DATABASE_PRIMARY")
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
        from_dt = getenv("FROM_DT")
        to_dt = getenv("TO_DT")

        # try:
        #     from_dt = datetime.strptime(from_dt, "%Y-%m-%dT%H:%M:%S")
        #     to_dt = datetime.strptime(to_dt, "%Y-%m-%dT%H:%M:%S")
        # except:
        #     from_dt = datetime.now().replace(hour=00, minute=00, second=00) - timedelta(days=1)
        #     to_dt = datetime.now().replace(hour=23, minute=59, second=59) - timedelta(days=1)

        if from_dt is None or to_dt is None:
            from_dt = datetime.now().replace(hour=00, minute=00, second=00) - timedelta(days=1)
            to_dt = datetime.now().replace(hour=23, minute=59, second=59) - timedelta(days=1)
        else:
            from_dt = datetime.strptime(from_dt, "%Y-%m-%dT%H:%M:%S.364Z")
            to_dt = datetime.strptime(to_dt, "%Y-%m-%dT%H:%M:%S.364Z")
        meta_data_dict_ret = infer_images_collection.find({
            'datetime_local': {
                '$gte': from_dt,
                '$lte': to_dt,
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
        temp = {'_uuid': online_ids, 'truck_list': object_list}
        data.update(temp)
        post_id = result_collection.insert_one(data)
    except Exception as e:
        logger.error("Error in dumping data | {}".format(e))
