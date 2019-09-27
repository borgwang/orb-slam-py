import pdb
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
        self.hist = {"pose": [], "points": []}

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

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.scam = pango.OpenGlRenderState(
            pango.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 1000),
            pango.ModelViewLookAt(0, -10, -40, 
                                  0, 0, 0, 
                                  0, -1, 0))

        self.handler = pango.Handler3D(self.scam)
        self.dcam = pango.CreateDisplay().SetBounds(
            pango.Attach(0), pango.Attach(1),
            pango.Attach(0), pango.Attach(1), -w/h).SetHandler(self.handler)

    def refresh(self, queue):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.dcam.Activate(self.scam)

        # draw points
        msg = queue.get()
        self.hist["points"].append(msg["points"])
        self.hist["pose"].append(msg["pose"])

        glPointSize(2)
        period = np.concatenate(self.hist["points"][:], axis=0)
        self.draw_points(period, colors=None)

        # draw cameras
        self.draw_cameras(self.hist["pose"])
        
        # finish frame
        pango.FinishFrame()

    def draw_points(self, points, colors=None):
        # set color 
        if colors is None:
            colors = [0.5, 0.5, 0.5]
        glColor4f(*colors, 0.2)

        glBegin(GL_POINTS)
        for p in points:
            glVertex3f(p[0], p[1], p[2])
        glEnd()

    def draw_line(self, points, colors=None):
        glBegin(GL_LINES)
        for i in range(len(points) - 1):
            glVertex3f(points[i, 0], points[i, 1], points[i, 2])
            glVertex3f(points[i+1, 0], points[i+1, 1], points[i+1, 2])
        glEnd()

    def draw_camera(self, pose, w=1.0, h_ratio=0.75, z_ratio=0.6):
        h = w * h_ratio
        z = w * z_ratio
        glPushMatrix()
        glMultTransposeMatrixd(pose)

        glBegin(GL_LINES)
        glColor3f(0, 0.6, 0)
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd()
        glPopMatrix()

    def draw_cameras(self, cameras, w=1.0, h_ratio=0.75, z_ratio=0.6):
        for c in cameras:
            self.draw_camera(c, w, h_ratio, z_ratio)

