import pdb
import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform


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


class FeatureExtractor(object):

    def __init__(self):
        self.prev = None
        self.orb = cv2.ORB_create()
        self.matcher  = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extract(self, frame):
        # feature extraction
        feats = cv2.goodFeaturesToTrack(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            4000, qualityLevel=0.01, minDistance=3)
        kp = [cv2.KeyPoint(*f[0], _size=3)  for f in feats]
        kp, des = self.orb.compute(frame, kp)

        # matching
        good = []
        if self.prev is not None:
            matches = self.matcher.knnMatch(des, self.prev[2], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    p1, p2 = self.prev[1][m.trainIdx].pt, kp[m.queryIdx].pt
                    p1, p2 = tuple(p1), tuple(p2)
                    good.append([p1, p2])
            good = np.asarray(good, dtype=int)
        self.prev = (frame, kp, des)
        return good


class PoseEstimator(object):

    def estimate(self, points):
        A_pts, B_pts = points[:, 0, :], points[:, 1, :]

        # normalize
        A_pts = normalize(A_pts)
        B_pts = normalize(B_pts)

        model, inliers = ransac((A_pts, B_pts),
                                EssentialMatrixTransform,
                                min_samples=8,
                                residual_threshold=0.001,
                                max_trials=1000)
        pose = self._recover_pose(model.params, A_pts, B_pts)

        A_pts = denormalize(A_pts[inliers])
        B_pts = denormalize(B_pts[inliers])

        points = np.stack([A_pts, B_pts], axis=1).astype(int)
        print("inliers: %d/%d" % (sum(inliers), len(points)))

        return points, pose

    @staticmethod
    def _recover_pose(E, A_pts, B_pts):
        _, R, t, _ = cv2.recoverPose(E, A_pts, B_pts)
        Rt = np.concatenate([R, t], axis=1)
        return Rt

