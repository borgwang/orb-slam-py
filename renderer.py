import time
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
            time.sleep(0.2)

    def init(self):
        w, h = 1024, 768
        pango.CreateWindowAndBind("main", w, h)
        glEnable(GL_DEPTH_TEST)

        self.scam = pango.OpenGlRenderState(
            pango.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 1000),
            pango.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0))

        self.handler = pango.Handler3D(self.scam)
        self.dcam = pango.CreateDisplay().SetBounds(
            pango.Attach(0), pango.Attach(1),
            pango.Attach(0), pango.Attach(1), -w/h).SetHandler(self.handler)

    def refresh(self, queue):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #glClearColor(1.0, 1.0, 1.0, 1.0)

        self.dcam.Activate(self.scam)

        # point cloud
        state = queue.get()
        self.hist.append(state)
        glPointSize(2)
        period = np.concatenate(self.hist[-40:], axis=0)
        self.draw_points(period, colors=None)

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

    def draw_line(self, points, colors=None):
        glBegin(GL_LINES)
        for i in range(len(points) - 1):
            glVertex3f(points[i, 0], points[i, 1], points[i, 2])
            glVertex3f(points[i+1, 0], points[i+1, 1], points[i+1, 2])
        glEnd()
