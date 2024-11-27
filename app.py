from flask import Flask, request, jsonify
from indicators import get_indicator
import json

app = Flask(__name__)

@app.route('/indicator', methods=['GET'])
def get_indicator_endpoint():
    symbol = request.args.get('symbol')
    interval = request.args.get('interval')
    indicator_name = request.args.get('indicator')
    kwargs = request.args.to_dict()
    kwargs.pop('symbol', None)
    kwargs.pop('interval', None)
    kwargs.pop('indicator', None)

    if not symbol or not interval or not indicator_name:
        return jsonify({'error': 'symbol, interval, and indicator parameters are required'}), 400

    try:
        data = get_indicator(symbol, interval, indicator_name, **kwargs)
        if data is None:
            return jsonify({'error': 'No data found for the given parameters'}), 404
        else:
            # Convert DataFrame to JSON
            data_json = data.reset_index().to_json(orient='records', date_format='iso')
            data_list = json.loads(data_json)
            return jsonify({'data': data_list}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)