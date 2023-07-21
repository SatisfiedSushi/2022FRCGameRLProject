import time


class Renderer:
    def __init__(self, environment):
        self.environment = environment
        self.done = False

    def start(self):
        self.done = False
        while not self.done and self.environment.resetted:
            self.environment.render()
            time.sleep(1)

    def stop(self):
        self.done = True
