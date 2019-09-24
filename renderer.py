from multiprocessing import Process, Queue

import numpy as np

import lib.pypangolin as pango
from OpenGL.GL import *


class Renderer(object):

    def __init__(self):
        self.scam, self.dcam = None, None
        self.handler = None
        self.process = None
        self.queue = Queue()
        self.hist = []

        p = Process(target=self.render, args=(self.queue,))
        p.daemon = True
        p.start()

    def render(self, queue):
        self.init()
        while not pango.ShouldQuit():
            self.refresh(queue)

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

    def refresh(self, queue):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #glClearColor(1.0, 1.0, 1.0, 1.0)

        self.dcam.Activate(self.scam)

        # point cloud
        state = queue.get()
        self.hist.append(state)
        glPointSize(1)
        self.draw_points(np.concatenate(self.hist, axis=0), colors=None)

        #pango.glDrawColouredCube()

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

