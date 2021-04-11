"""
Program saves screen captures for later use.
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


def collect_simple_data_with_model():
    """
    Using trained model to collect data. Data has all color channels untouched.
    """

    model = tf.keras.models.load_model("Model/trained_model")
    classes = ["cast", "pull", "wait"]

    # mss is fast
    sct = mss.mss()
    # set correct monitor location!
    monitor = {"top": 550, "left": 430, "width": 200, "height": 450}

    mouse = MouseController()

    frame_counter = 0  # used to name frames

    while True:
        time.sleep(0.1)  # probably not needed

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

        if pred_class == "cast":
            # saving to "cast" folder
            path_out = f"Unlabeled_Frames/cast/frame{frame_counter}.jpg"
            mouse.click(Button.right, 1)
            time.sleep(2.5)  # sleep while line is setting in game

        elif pred_class == "pull":
            # saving to "pull" folder
            path_out = f"Unlabeled_Frames/pull/frame{frame_counter}.jpg"
            mouse.click(Button.right, 1)
            time.sleep(2.5)  # sleep while line is setting in game

        else:
            # saving to "wait" folder
            # but not all wait pictures are needed

            if frame_counter % 40 == 0:
                path_out = f"Unlabeled_Frames/wait/frame{frame_counter}.jpg"

            else:
                path_out = f"Unlabeled_Frames/trash/frame{frame_counter}.jpg"

        frame = np.array(pil_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path_out, frame)

        frame = cv2.putText(frame, pred_class, (5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2,
                            cv2.LINE_AA)

        cv2.imshow('monitor', frame)
        cv2.waitKey(1)
        frame_counter += 1


def collect_stacked_data_with_model():
    """
    Uses old model to predict, but saves screen captures in stacked
    grayscale format.
    """
    model = tf.keras.models.load_model("Model/trained_model_1")
    classes = ["cast", "pull", "wait"]

    # mss is fast
    sct = mss.mss()
    # set correct monitor location!
    monitor = {"top": 550, "left": 430, "width": 200, "height": 450}

    mouse = MouseController()

    frame_counter = 0  # used to name frames

    # init stacked frame to empty
    # three grayscale pictures to replace color channels
    MAX_STACK_LEN = 3
    stacked_frames = deque([np.zeros((450, 200), dtype=np.uint8) for i in
                            range(MAX_STACK_LEN)], maxlen=MAX_STACK_LEN)

    # using this to clean stacked frame after catching a fish
    last_prediction = ""

    while True:
        time.sleep(0.1)  # probably not needed

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

        if pred_class == "cast":
            # saving to "cast" folder
            path_out = f"Unlabeled_Frames/cast/frame{frame_counter}.jpg"
            mouse.click(Button.right, 1)
            time.sleep(2.5)  # sleep while line is setting
            last_prediction = "cast"

        elif pred_class == "pull":
            # saving to "pull" folder
            path_out = f"Unlabeled_Frames/pull/frame{frame_counter}.jpg"
            mouse.click(Button.right, 1)
            time.sleep(2.5)  # sleep while line is setting
            last_prediction = "pull"

        else:
            last_prediction = "wait"
            # saving to "wait" folder
            # but not all wait pictures are needed

            if frame_counter % 40 == 0:
                path_out = f"Unlabeled_Frames/wait/frame{frame_counter}.jpg"

            else:
                path_out = f"Unlabeled_Frames/trash/frame{frame_counter}.jpg"

        frame_color = np.array(pil_img, dtype=np.uint8)  # shape (450, 200, 3)
        # shape (450, 200)
        frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

        stacked_frames.append(frame_gray)

        # Clean old floats from stack
        if last_prediction in ["cast", "pull"]:
            stacked_frames.append(frame_gray)
            stacked_frames.append(frame_gray)

        # stacked_frames to one frame
        # target shape (450, 200, 3)
        frame_from_stack = np.stack(stacked_frames, axis=2)

        cv2.imwrite(path_out, frame_from_stack)
        cv2.imshow('monitor0', stacked_frames[0])
        cv2.imshow('monitor1', stacked_frames[1])
        cv2.imshow('monitor2', stacked_frames[2])
        cv2.waitKey(1)
        frame_counter += 1


def main():
    """
    Collect data without previously trained model.
    Data has all color channels untouched.
    """
    # mss is fast
    sct = mss.mss()
    # set correct monitor location!
    monitor = {"top": 550, "left": 430, "width": 200, "height": 450}

    frame_counter = 0  # used to name frames

    while True:
        time.sleep(0.1)  # probably not needed
        frame = np.array(sct.grab(monitor))
        cv2.imshow('monitor', frame)
        cv2.imwrite(f"Unlabeled_Frames/frame{frame_counter}.jpg", frame)
        cv2.waitKey(1)
        frame_counter += 1


if __name__ == '__main__':
    main()
    # collect_simple_data_with_model()
    # collect_stacked_data_with_model()

