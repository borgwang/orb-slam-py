import pdb
import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform


np.set_printoptions(suppress=True)


W, H = 720, 1280
F = 240
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
Kinv = np.linalg.inv(K)


def add_ones(arr):
    return np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)


def normalize(pts):
    pts_ = add_ones(pts)
    ret = (Kinv @ pts_.T).T[:, :2]
    return ret


def denormalize(pts):
    pts_ = add_ones(pts)
    ret = (K @ pts_.T).T[:, :2]
    return ret


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

        # extract features
        point_pairs = self.extractor.extract(frame)
        if not len(point_pairs):
            return frame

        point_pairs[:, 0, :] = denormalize(point_pairs[:, 0, :])
        point_pairs[:, 1, :] = denormalize(point_pairs[:, 1, :])

        point_pairs = point_pairs.astype(int)
        # plot
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
                    p1, p2 = tuple(p1), tuple(p2)
                    good.append([p1, p2])

            good = np.asarray(good)

            # normalize
            good[:, 0, :] = normalize(good[:, 0, :])
            good[:, 1, :] = normalize(good[:, 1, :])

            model, inliers = ransac((good[:, 0], good[:, 1]),
                                    #FundamentalMatrixTransform,
                                    EssentialMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=0.01,
                                    max_trials=300)
            pose = self._extract_pose(model.params, good)
            print(pose)

            print("inliers: %d/%d" % (sum(inliers), len(good)))
            good = good[inliers]
        self.prev = (frame, kp, des)
        return good

    def _extract_pose(self, E, pts):
        _, R, t, _ = cv2.recoverPose(E, pts[:, 0, :], pts[:, 1, :])
        Rt = np.concatenate([R, t], axis=1)
        return Rt



def main():
    path = "./test2.mp4"
    extractor = Extractor()
    display = Display(path, extractor)
    display.draw()


if __name__ == "__main__":
    main()

