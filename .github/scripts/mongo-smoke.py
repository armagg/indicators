import os
import pandas as pd
from pymongo import MongoClient
import ta
from ta import store, ohlc
from ta.cache import ResearchCache, compute_cached
db=MongoClient("mongodb://mongo:27017").validation
db.client.admin.command("ping")
rows=[]
for i in range(100):
 t=pd.Timestamp("2025-01-01")+pd.Timedelta(hours=i)
 rows.append(dict(market_name="X",source="s",interval="1H",time=t.strftime("%Y-%m-%dT%H:%M:%S"),open=100+i,high=102+i,low=99+i,close=101+i,volume=10))
db.ohlc.insert_many(rows)
ohlc._client=db.client
os.environ.update(MONGO_URI="mongodb://mongo:27017",DB_NAME="validation",OHLC_COLLECTION_NAME="ohlc")
frame=ohlc.read_ohlc("X","s","1H")
assert len(frame)==100 and str(frame.index.tz)=="UTC"
result=ta.compute(frame,"rsi, macd:fast=8,slow=21,signal=5")
n=store.save(result,symbol="X",source="s",interval="1H",collection=db.values)
a=list(db.values.find({},{"_id":0,"computed_at":0}).sort([(k,1) for k in store.NATURAL_KEY]))
assert n>0 and len(a)==n
assert store.save(result,symbol="X",source="s",interval="1H",collection=db.values)==n
b=list(db.values.find({},{"_id":0,"computed_at":0}).sort([(k,1) for k in store.NATURAL_KEY]))
assert a==b and db.values.count_documents({})==n
cache=ResearchCache(db.columns,db.runs); cache.ensure_indexes()
x,s1=compute_cached(frame,"rsi",dataset="fixture",cache=cache)
y,s2=compute_cached(frame,"rsi",dataset="fixture",cache=cache)
assert s1=={"hits":0,"misses":1} and s2=={"hits":1,"misses":0}
pd.testing.assert_frame_equal(x,y,check_freq=False)
cache.put_run("job","fixture",{"a":1},{"status":"done"})
assert cache.get_run("job","fixture",{"a":1})=={"status":"done"}
print("Mongo OHLC, upsert, cache and status passed; documents",n)
