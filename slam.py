import pdb
import numpy as np
import cv2

from vo import FeatureExtractor
from vo import PoseEstimator


class Display(object):

    def __init__(self, path, extractor, pose_estimator):
        self.offsets = (200, 500)
        self.path = path

        self.extractor = extractor
        self.pose_estimator = pose_estimator

    def draw(self):
        # read video frame by frame
        vidcap = cv2.VideoCapture(self.path)
        success, frame = vidcap.read()

        window = cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("video", *self.offsets)

        while success:
            success, frame = vidcap.read()
            if success:
                frame = self.process_frame(frame)
                cv2.imshow("video", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        vidcap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        # preprocessing
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 1)

        # extract features
        point_pairs = self.extractor.extract(frame)
        if not len(point_pairs):
            return frame
        point_pairs, pose = self.pose_estimator.estimate(point_pairs)

        # plot
        for p1, p2 in point_pairs:
            cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 0))
            cv2.circle(frame, tuple(p2), 3, (0, 255, 0))
        return frame


def main():
    path = "./test2.mp4"
    extractor = FeatureExtractor()
    pose_estimator = PoseEstimator()
    display = Display(path, extractor, pose_estimator)
    display.draw()


if __name__ == "__main__":
    main()

