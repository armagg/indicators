from decouple import config
from pymongo import MongoClient
from redis import Redis

BASE_REDIS_KEY = "query_cache"


def get_ohlc_data(market_name, interval, start_time=None, end_time=None):
    cache_key = f'{BASE_REDIS_KEY}:{market_name}:{interval}:{start_time}:{end_time}'
    redis = Redis()
    cached_data = redis.get(cache_key)
    if cached_data is not None and cached_data != b'':
        return cached_data
    else:

        client = MongoClient(config("MONGO_URI"))
        db = client[config("DB_NAME")]
        collection = db[config("OHLC_COLLECTION_NAME")]

        query = {'market_name': symbol, 'interval': interval}
        if start_time and end_time:
            query['time'] = {'$gte': start_time, '$lte': end_time}

        cursor = collection.find(query)
        ohlc_data = pd.DataFrame(list(cursor))
        client.close()

        if ohlc_data.empty:
            return None

        ohlc_data['time'] = pd.to_datetime(ohlc_data['time'])
        ohlc_data.set_index('time', inplace=True)
        return ohlc_data