import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import lib.pypangolin as pango
from OpenGL.GL import *


def main():
    handler, s_cam, d_cam = init()
    while not pango.ShouldQuit():
        refresh(handler, s_cam, d_cam)


def init():
    win = pango.CreateWindowAndBind("pySimpleDisplay", 640, 480)
    glEnable(GL_DEPTH_TEST)

    pm = pango.ProjectionMatrix(640,480,420,420,320,240,0.1,1000);
    mv = pango.ModelViewLookAt(-0, 0.5, -3, 0, 0, 0, pango.AxisY)
    s_cam = pango.OpenGlRenderState(pm, mv)

    ui_width = 180
    handler=pango.Handler3D(s_cam)
    d_cam = pango.CreateDisplay().SetBounds(pango.Attach(0),
                                            pango.Attach(1),
                                            pango.Attach.Pix(ui_width),
                                            pango.Attach(1),
                                            -640.0/480.0).SetHandler(handler)
    return handler, s_cam, d_cam


def refresh(hander, s_cam, d_cam):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    d_cam.Activate(s_cam)
    pango.glDrawColouredCube()

    # point cloud
    points = np.random.random((10000, 3))
    colors = np.zeros((len(points), 3))
    colors[:, 1] = 1 - points[:, 0]
    colors[:, 2] = 1 - points[:, 1]
    colors[:, 0] = 1 - points[:, 2]
    points = points * 3 + 1
    glPointSize(3)
    draw_points(points, colors)

    pango.FinishFrame()


def draw_points(points, colors=None):
    if colors is None:
        colors = [[0.5, 0.5, 0.5] for _ in range(len(points))]
    glBegin(GL_POINTS)
    for p, c in zip(points, colors):
        glColor3f(c[0], c[1], c[2])
        glVertex3f(p[0], p[1], p[2])
    glEnd()


if __name__ == "__main__":
    main()
