"""

Velocity Generator

"""

import sys
import math
import numpy as np
import threading

class vel_generator:
    def __init__(self):
        self.vel = np.zeros((1, 3)) # [m/s]
        self.acc = np.zeros((1, 3)) # [m/s]
        self.vel_x  = 0
        self.vel_y = 0
        self.vel_z = 0
        self.vel_dir = 1

    # Random Velocity
    def RandAccGen(self):
        d_acc = np.random.randn(1, 3)
        self.acc = d_acc
        self.vel = self.vel + self.acc*0.5
        if self.vel[0, 0] > 0:
            self.vel_dir = 1
        else:
            self.vel_dir = 0
        self.vel_x = self.vel[0, 0]
        self.vel_y = self.vel[0, 1]
        self.vel_z = self.vel[0, 2]

        # print("velocity: {}".format(self.vel))
        threading.Timer(0.01, self.RandAccGen).start()

def main():
    vel_1 = vel_generator()
    vel_1.RandAccGen()
    print(velocity)


if __name__ == '__main__':
    main()
    sys.setrecursionlimit(30000)
