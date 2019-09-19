import pdb
import numpy as np
import cv2
import os


class Display(object):

    def __init__(self, path):
        self.offsets = (200, 500)
        self.path = path

        self.prev = None

    def draw(self):
        # read video frame by frame
        vidcap = cv2.VideoCapture(self.path)
        _, frame = vidcap.read()

        window = cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("video", *self.offsets)

        orb = cv2.ORB_create()
        bf_matcher  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        while vidcap.isOpened():
            success, frame = vidcap.read()
            if success:
                frame = self.process_frame(frame, orb, bf_matcher)
                cv2.imshow("video", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        vidcap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, orb, matcher):
        # preprocessing
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 1)

        # feature extraction
        feats = cv2.goodFeaturesToTrack(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            1000, qualityLevel=0.01, minDistance=3)
        kp = [cv2.KeyPoint(*f[0], _size=3)  for f in feats]
        kp, des = orb.compute(frame, kp)
        frame_ = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0))

        # matching
        if self.prev is not None:
            matches = matcher.match(des, self.prev[2])
            for m in matches:
                print(m.trainIdx, m.queryIdx)
                p1, p2 = self.prev[1][m.trainIdx].pt, kp[m.queryIdx].pt
                p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
                cv2.line(frame_, p1, p2, (0, 255, 0), 1)
        self.prev = (frame, kp, des)
        return frame_


def main():
    path = "./test.mp4"
    display = Display(path)
    display.draw()


if __name__ == "__main__":
    main()
