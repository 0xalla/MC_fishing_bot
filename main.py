"""
Main file for actually testing and using trained model.
"""


import time

# for neural network
import tensorflow as tf
from tensorflow import keras

# for pictures
import mss
import numpy as np
import cv2
from PIL import Image

# for controlling mouse
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

# for fishing with stacked model
from collections import deque


def test_model():
    model = tf.keras.models.load_model("train_model/Model/last_trained_model")
    classes = ["cast", "pull", "wait"]

    filename = "PATH HERE"

    img = keras.preprocessing.image.load_img(
        filename, target_size=(450, 200)
    )

    keras_frame = keras.preprocessing.image.img_to_array(img)
    frame_batch = tf.expand_dims(keras_frame, 0)

    prediction = model.predict(frame_batch)
    pred_class = classes[np.argmax(prediction)]  # predicted class
    print("prediction:", prediction)
    print("prediction class:", pred_class)

    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.putText(frame, pred_class, (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)

    cv2.imshow('test_frame', frame)
    cv2.waitKey(0)


def fishing_with_stacked_model():
    """
    Fishing with stacked model (color channels are replaced with grayscale
    pictures).
    """
    model = tf.keras.models.load_model("train_model/Model/stacked_model")
    classes = ["cast", "pull", "wait"]

    # mss is fast
    sct = mss.mss()
    # set correct monitor location !
    monitor = {"top": 550, "left": 430, "width": 200, "height": 450}

    mouse = MouseController()

    # init stacked frame to empty
    # three grayscale pictures to replace color channels
    MAX_STACK_LEN = 3
    stacked_frames = deque([np.zeros((450, 200), dtype=np.uint8) for i in
                            range(MAX_STACK_LEN)], maxlen=MAX_STACK_LEN)

    # using this to clean stacked frame after catching a fish
    last_prediction = ""

    while True:
        t_1 = time.time()
        # using little sleep to keep reaction time close to human
        time.sleep(0.1)

        # python-mss.readthedocs.io/examples.html
        sct_img = sct.grab(monitor)
        pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw",
                                  "BGRX")

        frame_color = np.array(pil_img, dtype=np.uint8)  # shape (450, 200, 3)
        frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        stacked_frames.append(frame_gray)

        # clean old floats from stack
        if last_prediction in ["cast", "pull"]:
            stacked_frames.append(frame_gray)
            stacked_frames.append(frame_gray)

        # stacked_frames to one frame
        # target shape (450, 200, 3)
        frame_from_stack = np.stack(stacked_frames, axis=2)
        keras_frame = keras.preprocessing.image.img_to_array(frame_from_stack)
        frame_batch = tf.expand_dims(keras_frame, 0)
        prediction = model.predict(frame_batch)
        pred_class = classes[np.argmax(prediction)]

        print()
        print(pred_class)

        cv2.imshow('monitor', frame_from_stack)
        cv2.waitKey(1)
        print("fps:", 1 // (time.time() - t_1))

        if pred_class == "cast":
            mouse.click(Button.right, 1)
            time.sleep(2.5)  # sleep while line is setting in game
            last_prediction = "cast"

        elif pred_class == "pull":
            mouse.click(Button.right, 1)
            time.sleep(2.5)  # sleep while line is setting in game
            last_prediction = "pull"

        else:
            # no need for action, just wait
            last_prediction = "wait"


def main():
    """
    Fishing with simple model (color channels are untouched).
    """
    model = tf.keras.models.load_model("train_model/Model/trained_model")
    classes = ["cast", "pull", "wait"]

    # mss is fast
    sct = mss.mss()
    # set correct monitor location !
    monitor = {"top": 550, "left": 430, "width": 200, "height": 450}

    mouse = MouseController()

    while True:
        t_1 = time.time()
        # using little sleep to keep reaction time close to human
        time.sleep(0.1)

        # python-mss.readthedocs.io/examples.html
        sct_img = sct.grab(monitor)
        pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw",
                                  "BGRX")
        keras_frame = keras.preprocessing.image.img_to_array(pil_img)
        frame_batch = tf.expand_dims(keras_frame, 0)
        prediction = model.predict(frame_batch)
        pred_class = classes[np.argmax(prediction)]

        print()
        print(pred_class)

        frame = np.array(pil_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.putText(frame, pred_class, (5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)

        cv2.imshow('monitor', frame)
        cv2.waitKey(1)
        print("fps:", 1 // (time.time() - t_1))

        if pred_class == "cast":
            mouse.click(Button.right, 1)
            time.sleep(2.5)  # sleep while line is setting in game

        elif pred_class == "pull":
            mouse.click(Button.right, 1)
            time.sleep(2.5)  # sleep while line is setting in game

        else:
            # no need for action, just wait
            pass


if __name__ == '__main__':
    main()
    # test_model()
    # fishing_with_stacked_model()
