import pdb
import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


class Display(object):

    def __init__(self, path, extractor):
        self.offsets = (200, 500)
        self.path = path

        self.extractor = extractor

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

        point_pairs = self.extractor.extract(frame)
        for p1, p2 in point_pairs:
            cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 0))
            cv2.circle(frame, tuple(p2), 3, (0, 255, 0))
        return frame


class Extractor(object):

    def __init__(self):
        self.prev = None

        self.orb = cv2.ORB_create()
        self.matcher  = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extract(self, frame):
        # feature extraction
        feats = cv2.goodFeaturesToTrack(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            3600, qualityLevel=0.01, minDistance=3)
        kp = [cv2.KeyPoint(*f[0], _size=3)  for f in feats]
        kp, des = self.orb.compute(frame, kp)

        # matching
        good = []
        if self.prev is not None:
            matches = self.matcher.knnMatch(des, self.prev[2], k=2)
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    p1, p2 = self.prev[1][m.trainIdx].pt, kp[m.queryIdx].pt
                    p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
                    good.append([p1, p2])

            good = np.asarray(good)
            # fundamental matrix estimation
            model, inliers = ransac((good[:, 0], good[:, 1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=300)
            print("inliers: %d/%d" % (sum(inliers), len(good)))
            good = good[inliers]
        self.prev = (frame, kp, des)
        return good


def main():
    path = "./test.mp4"
    extractor = Extractor()
    display = Display(path, extractor)
    display.draw()


if __name__ == "__main__":
    main()

