import numpy as np
import cv2
import os


class Display(object):

    def __init__(self, path):
        self.offsets = (200, 500)
        self.path = path
        self.shape = None

    def draw(self):
        # read video frame by frame
        vidcap = cv2.VideoCapture(self.path)
        _, frame = vidcap.read()
        self.shape = frame.shape

        window = cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("video", *self.offsets)

        while vidcap.isOpened():
            success, frame = vidcap.read()
            if success:
                # transpose frame
                frame = cv2.transpose(frame)
                cv2.imshow("video", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        vidcap.release()
        cv2.destroyAllWindows()


def main():
    path = "./test.mp4"
    display = Display(path)
    display.draw()


if __name__ == "__main__":
    main()
