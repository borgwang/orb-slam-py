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

        orb = cv2.ORB_create()
        while vidcap.isOpened():
            success, frame = vidcap.read()
            if success:
                frame = self.process_frame(orb, frame)
                cv2.imshow("video", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        vidcap.release()
        cv2.destroyAllWindows()

    def process_frame(self, orb, frame):
        # rotate frame
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 1)

        # orb feature
        kp, des = orb.detectAndCompute(frame, None)
        frame = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0))

        return frame


def main():
    path = "./test.mp4"
    display = Display(path)
    display.draw()


if __name__ == "__main__":
    main()
