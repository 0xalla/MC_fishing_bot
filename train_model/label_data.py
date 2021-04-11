"""
Tool to make data labeling faster.

some parts are copy from:
https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
"""


import time
import os
from os.path import isfile, join
import cv2
import readchar


def main():
    path_in = "Unlabeled_Frames/"
    files = [f for f in os.listdir(path_in) if isfile(join(path_in, f))]

    files.sort(key=lambda x: int(x[5:-4]))

    for filename in files:
        time.sleep(0.1)  # practical sleep
        print(filename)
        frame = cv2.imread(path_in + filename, cv2.IMREAD_COLOR)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        key_press = readchar.readkey()
        print(key_press)

        if key_press == 'a':
            # no need to do anything (wait)
            path_out = "Labeled_Frames/wait/" + filename

        elif key_press == 's':
            # cast line (cast)
            path_out = "Labeled_Frames/cast/" + filename

        elif key_press == 'd':
            # fish in line (pull)
            path_out = "Labeled_Frames/pull/" + filename

        else:
            # put frame to trash
            path_out = "Unlabeled_Frames/trash/" + filename

        cv2.imwrite(path_out, frame)


if __name__ == '__main__':
    main()
