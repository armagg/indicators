
def compute_macd(data, fast=12, slow=26, signal=9):
    macd = ta.macd(data['close'], fast=fast, slow=slow, signal=signal)
    data = data.join(macd)
    return data
