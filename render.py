from multiprocessing import Process, Queue
import matplotlib.pyplot as plt

import numpy as np


class MapRenderer(object):

    def __init__(self):
        self.render_process = None

    def start(self):
        self.render_process = Process(target=self.render, args=())
        self.render_process.start()
        self.render_process.join()

    def render(self):
        self.init()
        while True:
            self.refresh()

    def init(self):
        pass

    def refresh(self):
        pass


if __name__ == "__main__":
    map_renderer = MapRenderer()
    map_renderer.start()
