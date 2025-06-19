from flask import Flask, request, jsonify
from features.autotrade import run_autotrader

app = Flask(__name__)


@app.route("/trade")
def trade():
    symbol = request.args.get("symbol", "AAPL")
    run_autotrader([symbol])
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
