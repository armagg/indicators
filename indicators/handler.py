# indicators.py

import pandas as pd
import pandas_ta as ta
from pymongo import MongoClient
from redis import Redis

# Initialize Redis client
redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)




def get_indicator(symbol, interval, indicator_name, **kwargs):
    # Check cache first
    cache_key = f"{symbol}:{interval}:{indicator_name}:{kwargs}"
    cached_data = redis_client.get(cache_key)
    if cached_data:
        print("Fetching data from cache")
        df = pd.read_json(cached_data, orient='split')
        return df

    # Fetch OHLC data
    data = get_ohlc_data(symbol, interval)
    if data is None:
        return None

    # Compute indicator
    if indicator_name == 'moving_average':
        window = kwargs.get('window', 20)
        data = compute_moving_average(data, window)
    elif indicator_name == 'rsi':
        period = kwargs.get('period', 14)
        data = compute_rsi(data, period)
    elif indicator_name == 'macd':
        fast = kwargs.get('fast', 12)
        slow = kwargs.get('slow', 26)
        signal = kwargs.get('signal', 9)
        data = compute_macd(data, fast, slow, signal)
    else:
        raise ValueError(f"Indicator {indicator_name} not recognized.")

    # Cache the result
    redis_client.setex(cache_key, CACHE_EXPIRATION, data.to_json(orient='split'))

    # Save data to MongoDB asynchronously
    from tasks import save_indicator_data  # Import here to avoid circular import
    save_indicator_data.delay(symbol, interval, indicator_name, data.reset_index().to_dict(orient='records'))

    return data