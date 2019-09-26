from multiprocessing import Queue
import pdb
import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from skimage.transform import FundamentalMatrixTransform

from renderer import Renderer

np.set_printoptions(suppress=True)

W, H = 720, 1280
F = 360
K = np.array([
    [F, 0, W // 2],
    [0, F, H // 2],
    [0, 0, 1]])


class Frame(object):

    def __init__(self, frame):
        self._data = frame
        self.idx = None
        self.pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]).astype(float)

        self._features = None
        self._discriptor = None
        self.extract(frame)


    def extract(self, frame):
        # extract keypoints and dicriptor
        orb = cv2.ORB_create()
        feats = cv2.goodFeaturesToTrack(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            3000, qualityLevel=0.01, minDistance=3)
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

    def match_frames(self, frame_pair):
        curr, prev = frame_pair
        # matching two frames
        good = []
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
            curr.discriptor, prev.discriptor, k=2)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                p1 = curr.features[m.queryIdx].pt
                p2 = prev.features[m.trainIdx].pt
                p1, p2 = tuple(p1), tuple(p2)
                good.append([p1, p2])
        good = np.asarray(good, dtype=int)
        return good

    def reconstruct3d(self):
        # extrct keypoints and match
        if self.size < 2:
            return None, None
        frame_pair = (self._frames[-1], self._frames[-2])
        point_pairs = self.match_frames(frame_pair)

        # estimate points
        point_pairs, points3d = self.pose_estimator.estimate(point_pairs, frame_pair)

        # add to points to map
        self.renderer.queue.put(points3d)

        return point_pairs, points3d

    def estimate_pose(self):
        pass


class PoseEstimator(object):

    @property
    def It(self):
        It = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        return It.astype(float)

    def estimate(self, points, frames):
        A_pts, B_pts = points[:, 0, :], points[:, 1, :]
        curr, prev = frames

        # normalize
        A_pts = self.normalize(A_pts)
        B_pts = self.normalize(B_pts)

        model, inliers = ransac((A_pts, B_pts),
                                EssentialMatrixTransform,
                                #FundamentalMatrixTransform,
                                min_samples=8,
                                residual_threshold=0.001,
                                max_trials=300)

        Rt = self.extract_Rt(model.params, A_pts, B_pts)

        # triangulate ot get 3D points
        curr.pose = Rt @ prev.pose
        points3d = self.triangulate2(curr.pose, prev.pose, A_pts[inliers], B_pts[inliers])
        print(points3d.shape)
        print(prev.pose)
        print(curr.pose)
        #pdb.set_trace()

        # denormalize
        A_pts = self.denormalize(A_pts[inliers])
        B_pts = self.denormalize(B_pts[inliers])

        point_pairs = np.stack([A_pts, B_pts], axis=1).astype(int)
        print("inliers: %d/%d" % (sum(inliers), len(points)))
        return point_pairs, points3d

    @staticmethod
    def extract_Rt(E, pts1, pts2):
        points, R, t, mask = cv2.recoverPose(E, pts1, pts2)
        ret = np.eye(4)
        R_t = np.concatenate([R, t], axis=1)
        ret[:3] = R_t
        return ret

    @staticmethod
    def triangulate(pose1, pose2, points1, points2):
        pose1, pose2 = pose1[:3], pose2[:3]
        pts4d_hom = cv2.triangulatePoints(pose1, pose2, points1.T, points2.T).T
        pts4d = pts4d_hom / pts4d_hom[:, 3:]
        good_pts4d = pts4d[pts4d[:, 2] > 0]
        print("%s/%s" % (len(good_pts4d), len(pts4d)))
        return good_pts4d

    @staticmethod
    def triangulate2(pose1, pose2, points1, points2):
        ret = np.zeros((points1.shape[0], 4))
        pose1 = np.linalg.inv(pose1)
        pose2 = np.linalg.inv(pose2)
        for i, p in enumerate(zip(points1, points2)):
            A = np.zeros((4, 4))
            A[0] = p[0][0] * pose1[2] - pose1[0]
            A[1] = p[0][1] * pose1[2] - pose1[1]
            A[2] = p[1][0] * pose2[2] - pose2[0]
            A[3] = p[1][1] * pose2[2] - pose2[1]
            _, _, vt = np.linalg.svd(A)
            ret[i] = vt[3]
        ret /= ret[:, 3:]
        ret = ret[ret[:, 2] > 0]
        return ret

    @staticmethod
    def _add_ones(arr):
        return np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)

    def normalize(self, pts):
        pts_ = self._add_ones(pts)
        Kinv = np.linalg.inv(K)
        ret = (Kinv @ pts_.T).T[:, :2]
        return ret

    def denormalize(self, pts):
        pts_ = self._add_ones(pts)
        ret = (K @ pts_.T).T[:, :2]
        return ret

