from .utils import predict

from flask import Flask
from flask import request, jsonify

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict() -> float:
    flight_info = request.get_json()

    y_pred = predict.predict_price(flight_info)

    result = {"price_predicted": float(y_pred)}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
