from multiprocessing import Queue
import pdb
import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

from renderer import Renderer

np.set_printoptions(suppress=True)

W, H = 720, 1280
F = 240
K = np.array([
    [F, 0, W // 2],
    [0, F, H // 2],
    [0, 0, 1]])
Kinv = np.linalg.inv(K)


class Frame(object):

    def __init__(self, frame):
        self._data = frame
        self.idx = None

        self._features = None
        self._discriptor = None
        self.extract(frame)

    def extract(self, frame):
        # extract keypoints and dicriptor
        orb = cv2.ORB_create()
        feats = cv2.goodFeaturesToTrack(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            4000, qualityLevel=0.01, minDistance=3)
        kp = [cv2.KeyPoint(*f[0], _size=3)  for f in feats]
        kp, des = orb.compute(frame, kp)
        self._features = kp
        self._discriptor = des

    @property
    def data(self):
        return self._data

    @property
    def features(self):
        return self._features

    @property
    def discriptor(self):
        return self._discriptor


class FrameManager(object):

    def __init__(self):
        self._frames = []
        self.pose_estimator = PoseEstimator()
        self.renderer = Renderer()

    def add(self, frame):
        frame.idx = self.size
        self._frames.append(frame)

    @property
    def size(self):
        return len(self._frames)

    @property
    def frames(self):
        return self._frames

    def match_frames(self, frame1, frame2):
        # matching two frames
        good = []
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
            frame1.discriptor, frame2.discriptor, k=2)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                p1 = frame1.features[m.queryIdx].pt
                p2 = frame2.features[m.trainIdx].pt
                p1, p2 = tuple(p1), tuple(p2)
                good.append([p1, p2])
        good = np.asarray(good, dtype=int)
        return good

    def reconstruct3d(self):
        # extrct keypoints and match
        if self.size < 2:
            return None, None
        point_pairs = self.match_frames(self._frames[-1], self._frames[-2])
        # estimate points
        point_pairs, points3d = self.pose_estimator.estimate(point_pairs)

        # TODO: add frame to map
        self.renderer.queue.put(points3d)
        return point_pairs, points3d


class PoseEstimator(object):

    @property
    def pose1(self):
        pose = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0]])
        return pose.astype(float)

    def estimate(self, points):
        A_pts, B_pts = points[:, 0, :], points[:, 1, :]

        # normalize
        A_pts = self.normalize(A_pts)
        B_pts = self.normalize(B_pts)

        model, inliers = ransac((A_pts, B_pts),
                                EssentialMatrixTransform,
                                min_samples=8,
                                residual_threshold=0.001,
                                max_trials=300)
        Rt = self.extract_Rt(model.params, B_pts, A_pts)

        # triangulate ot get 3D points
        points3d = self.triangulate(Rt, self.pose1, A_pts.T, B_pts.T)

        # denormalize
        A_pts = self.denormalize(A_pts[inliers])
        B_pts = self.denormalize(B_pts[inliers])

        point_pairs = np.stack([A_pts, B_pts], axis=1).astype(int)
        print("inliers: %d/%d" % (sum(inliers), len(points)))
        return point_pairs, points3d

    @staticmethod
    def extract_Rt(E, pts1, pts2):
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2)
        R_t = np.concatenate([R, t], axis=1)
        return R_t

    @staticmethod
    def triangulate(pose1, pose2, points1, points2):
        pts4d = cv2.triangulatePoints(pose1, pose2, points1, points2).T
        good_pts4d = pts4d[pts4d[:, 3] > 0.003]
        print("%s/%s" % (len(good_pts4d), len(pts4d)))
        good_pts4d /= good_pts4d[:, 3:]
        return good_pts4d[:, :3]

    @staticmethod
    def _add_ones(arr):
        return np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)

    def normalize(self, pts):
        pts_ = self._add_ones(pts)
        ret = (Kinv @ pts_.T).T[:, :2]
        return ret

    def denormalize(self, pts):
        pts_ = self._add_ones(pts)
        ret = (K @ pts_.T).T[:, :2]
        return ret

