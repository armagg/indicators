
def compute_rsi(data, period=14):
    data['RSI'] = ta.rsi(data['close'], length=period)
    return data
