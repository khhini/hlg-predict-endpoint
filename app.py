from ast import excepthandler
from flask import Flask, request
from utilities import prediction
import json

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello_world():
    return json.dumps({'ping':'pong'})

@app.route("/", methods=["POST"])
def post_requst():
    req = request.get_json()
    im_b64 = req["img"]
    res = prediction(im_b64)
    return json.dumps(res), 200

