# tasks.py

from celery import Celery
from pymongo import MongoClient

celery_app = Celery('tasks', broker=f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}')

@celery_app.task
def save_indicator_data(symbol, interval, indicator_name, data):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[INDICATORS_COLLECTION_NAME]

    # Prepare data for insertion
    for item in data:
        item['symbol'] = symbol
        item['interval'] = interval
        item['indicator'] = indicator_name

    # Insert data into MongoDB
    if data:
        collection.insert_many(data)
    client.close()