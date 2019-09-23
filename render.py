from multiprocessing import Process
import time
import pdb
import numpy as np

import OpenGL.GL as gl
import pypangolin as pgl


class MapRenderer(object):

    def __init__(self):
        self.render_process = None
        self.scam, self.dcam = None, None

    def start(self):
        self.render_process = Process(target=self.render, args=())
        self.render_process.start()
        self.render_process.join()

    def render(self):
        self.init()
        while 1:
            self.refresh()

    def init(self):
        W, H = 640, 480
        pgl.CreateWindowAndBind("main", W, H)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pgl.OpenGlRenderState(
            pgl.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
            pgl.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pgl.AxisDirection.AxisY))

        self.dcam = pgl.CreateDisplay()
        self.dcam.SetBounds(pgl.Attach(0.0), pgl.Attach(1.0),
                       pgl.Attach(0.0), pgl.Attach(1.0), -640.0/480.0)
        self.dcam.SetHandler(pgl.Handler3D(self.scam))

    def refresh(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        # point cloud
        points = np.random.random((10000, 3)) * 3 - 1
        gl.glPointSize(1)
        gl.glColor3f(1.0, 0.0, 0.0)
        self.draw_points(points)

        # point cloud
        points = np.random.random((10000, 3))
        colors = np.zeros((len(points), 3))
        colors[:, 1] = 1 - points[:, 0]
        colors[:, 2] = 1 - points[:, 1]
        colors[:, 0] = 1 - points[:, 2]
        points = points * 3 + 1
        gl.glPointSize(1)
        self.draw_points(points, colors)

        # draw camera
        pose = np.identity(4)
        pose[:3, 3] = np.random.randn(3)
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 1.0)

        # finish frame
        pgl.FinishFrame()

    def draw_points(self, points, colors=None):
        if colors is None:
            colors = [[0.5, 0.5, 0.5] for _ in range(len(points))]
        gl.glBegin(gl.GL_POINTS)
        for p, c in zip(points, colors):
            gl.glColor3f(c[0], c[1], c[2])
            gl.glVertex3f(p[0], p[1], p[2])
        gl.glEnd()


if __name__ == "__main__":
    import sys

    def trace(frame, event, arg):
        print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
        return trace

    #sys.settrace(trace)
    map_renderer = MapRenderer()
    map_renderer.start()

