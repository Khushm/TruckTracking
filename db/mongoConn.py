from pymongo import MongoClient
from loguru import logger
from os import getenv
from urllib.parse import quote_plus


user = getenv("MONGO_USERNAME_PRIMARY")
password = getenv("MONGO_PASSWORD_PRIMARY")
host = str(getenv("MONGO_HOST_PRIMARY"))
db = getenv("MONGO_DATABASE_PRIMARY")
MONGO_URL = "mongodb://%s:%s@%s" % (quote_plus(user), quote_plus(password), host)
mongo_client = None
# connect to db
def get_mongo_client():
    try:
        global mongo_client
        mongo_client = MongoClient(MONGO_URL)
        mongo_client.admin.authenticate(user, password)
        database = mongo_client[db]
        return database
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
def fetch_data(data_col, from_dt, to_dt, camera, panel):
    try:
        meta_data_dict_ret = data_col.find({
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
def load_meta_data(meta_data_col, ai_id=4):
    try:
        meta_data_dict_ret = meta_data_col.find({
            'ai_id': ai_id
        })
        return meta_data_dict_ret
    except Exception as e:
        logger.error(e)


# push results back to db
def push_data(mongo_coll, online_ids, data):
    try:
        try:
            data['prev_mongo_id'] = data['_id']
            del data['_id']
        except:
            pass

        temp = {}
        temp['_uuid'] = online_ids
        data.update(temp)

        post_id = mongo_coll.insert_one(data)

        # post_id = mongo_coll.insert_one(Truckdata).inserted_id
        # logger.info("Data Inserted!")
    except Exception as e:
        logger.error("Error in dumping data | {}".format(e))
