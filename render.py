from multiprocessing import Process

import numpy as np

import pypangolin as pango
from OpenGL.GL import *


class Renderer(object):

    def __init__(self):
        self.scam, self.dcam = None, None
        self.handler = None
        self.process = None

    def run(self):
        # init process
        self.process = Process(target=self.render, args=())
        self.process.start()
        self.process.join()

    def render(self):
        self.init()
        while not pango.ShouldQuit():
            self.refresh()

    def init(self):
        pango.CreateWindowAndBind("main", 640, 480)
        glEnable(GL_DEPTH_TEST)

        self.scam = pango.OpenGlRenderState(
            pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
            pango.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pango.AxisY))

        self.handler = pango.Handler3D(self.scam)
        self.dcam = pango.CreateDisplay().SetBounds(
            pango.Attach(0), pango.Attach(1),
            pango.Attach(0), pango.Attach(1), -640.0/480.0).SetHandler(self.handler)

    def refresh(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)
        pango.glDrawColouredCube()

        ## point cloud
        points = np.random.random((10000, 3))
        colors = np.zeros((len(points), 3))
        colors[:, 1] = 1 - points[:, 0]
        colors[:, 2] = 1 - points[:, 1]
        colors[:, 0] = 1 - points[:, 2]
        points = points * 3 + 1
        glPointSize(3)
        self.draw_points(points, colors)

        # finish frame
        pango.FinishFrame()

    def draw_points(self, points, colors=None):
        if colors is None:
            colors = [[0.5, 0.5, 0.5] for _ in range(len(points))]
        glBegin(GL_POINTS)
        for p, c in zip(points, colors):
            glColor3f(c[0], c[1], c[2])
            glVertex3f(p[0], p[1], p[2])
        glEnd()


if __name__ == "__main__":
    Renderer().run()

