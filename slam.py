import numpy as np
import cv2

from vo import Frame, FrameManager


class FrameDisplay(object):

    def __init__(self, path):
        self.offsets = (200, 500)
        self.path = path

        self.frame_manager = FrameManager()

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
        # frame preprocessing
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame, 1)

        self.frame_manager.add(Frame(frame))
        point_pairs, points3d = self.frame_manager.reconstruct3d()
        frame = self.frame_manager.frames[-1].data
        if point_pairs is None:
            return frame

        # plot
        for p1, p2 in point_pairs:
            cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 0))
            cv2.circle(frame, tuple(p2), 3, (0, 255, 0))
        return frame


def main():
    path = "./test4.mp4"
    display = FrameDisplay(path)
    display.draw()


if __name__ == "__main__": 
    main()

