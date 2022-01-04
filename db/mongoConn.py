from pymongo import MongoClient
from loguru import logger
from os import getenv
from urllib.parse import quote_plus


user = getenv("MONGO_USERNAME")
password = getenv("MONGO_PASSWORD")
host = str(getenv("MONGO_HOST"))
db = getenv("MONGO_DATABASE")
coll = getenv("MONGO_COLLECTION")

MONGO_URL = "mongodb://%s:%s@%s" % (quote_plus(user), quote_plus(password), host)


def get_mongo_client():
    try:
        mongo_client = MongoClient(MONGO_URL)
        # return mongo_client
        mongo_client.admin.authenticate(user, password)
        database = mongo_client[db]
        collection = database[coll]
        return collection
    except Exception as e:
        logger.debug(f'Error while Connecting to Mongo Client: | Error:{e}')
        raise e
