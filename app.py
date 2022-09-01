import re
from flask import Flask, jsonify, json
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from os.path import dirname, join
import detect_mask_video as dmv

# from detect_mask_video import detect_and_predict_mask;

app = Flask(__name__)


@app.route('/')
def index():

    # # load our serialized face detector model from disk
    prototxtPath = join(dirname(__file__), "face_detector/deploy.prototxt")
    weightsPath = join(dirname(__file__),
                       "face_detector/res10_300x300_ssd_iter_140000.caffemodel")

    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("mask_detector.model")

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    # loop over the frames from the video stream
    i=0
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        img = cv2.imread("/dataset/without_mask/289.jpg")
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = dmv.detect_and_predict_mask(img, faceNet, maskNet)
        i+=1
        yield(i)
    return 124


if __name__ == "__main__":
    app.run(debug=True)
