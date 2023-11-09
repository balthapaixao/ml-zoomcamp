from .utils import predict_utilities

from flask import Flask
from flask import request, jsonify

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    flight_info = request.get_json()
    y_pred = predict_utilities.predict_price(flight_info)
    print(f"Price predicted: ${int(y_pred)}.00")

    result = {"price_predicted": int(y_pred)}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
