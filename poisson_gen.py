'''
    Simple 2D Poisson Disk Sampling

    Ref:
    https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
    Bridson, R. (2007, August). Fast Poisson disk sampling in arbitrary dimensions. In SIGGRAPH sketches (p. 22).
'''

import math
import random
import numpy as np
import matplotlib.pyplot as plt

class Poisson2D:
    def __init__(self, sep):
        self.__size = 1
        self.__sep = sep
        self.__dx = sep / math.sqrt(2)
        self.__ngrid = math.floor(self.__size / self.__dx)

        self.grid = np.empty((self.__ngrid * self.__ngrid, 2), dtype=float)
        self.grid.fill(-1)
        self.active = []
        self.index_all = []

    def generate(self):
        # initial starting from the center
        x0 = self.__size * 0.5
        y0 = self.__size * 0.5
        i_row = math.floor(x0 / self.__dx)
        i_col = math.floor(y0 / self.__dx)
        index = self.__coord2Index(i_row, i_col)
        self.grid[index,0] = x0
        self.grid[index,1] = y0
        self.index_all.append(index)
        self.active.append(index)

        # start random sampling
        while len(self.active) > 0:
            self.__newPts()

        self.draw()

    def __newPts(self):
        # get a random active points
        randIndex = math.floor(random.uniform(0, len(self.active)))
        refIndex = self.active[randIndex]
        ref_pos = self.grid[refIndex]
        flag_found = False
        # try to sample at most 30 points
        for i_trial in range(30):
            # trail position
            radius = random.uniform(self.__sep, 2*self.__sep)
            angle = random.uniform(0, 2*math.pi)
            trial_pos = ref_pos + np.array([radius * math.cos(angle), radius * math.sin(angle)], dtype=float)
            # check trial position
            trial_row = math.floor(trial_pos[0] / self.__dx)
            trial_col = math.floor(trial_pos[1] / self.__dx)
            trial_index = self.__coord2Index(trial_row, trial_col)

            if not self.__isInside(trial_row, trial_col):
                continue
            # check if inside target shape
            # if not self.__isInRectangle(trial_pos[0], trial_pos[1]):
            #     continue
            # if not self.__isInCircle(trial_pos[0], trial_pos[1]):
            #     continue
            if self.__isOccupied(trial_index):
                continue
            
            # check the neighbor of the trial position
            flag_good_pos = self.__isNeighborGood(trial_row, trial_col, trial_pos)

            if flag_good_pos:
                self.grid[trial_index] = trial_pos
                self.index_all.append(trial_index)
                self.active.append(trial_index)
                flag_found = True
        # if no such point can be found, remove the reference point
        if not flag_found:
            self.active.pop(randIndex)
    
    def __coord2Index(self, i, j):
        return i + j * self.__ngrid

    def __isOccupied(self, index):
        if self.grid[index,0] == -1 and self.grid[index,1] == -1:
            return False
        else:
            return True

    def __isInside(self, row, col):
        if row < self.__ngrid and row >= 0 and col < self.__ngrid and col >= 0:
            return True
        else:
            return False        
    
    def __isNeighborGood(self, row, col, pos):
        for i in range(-1,2):
            for j in range(-1,2):
                if self.__isInside(row+i, col+j):
                    check_index = self.__coord2Index(row+i, col+j)
                    if self.__isOccupied(check_index):
                        dist = np.linalg.norm(pos - self.grid[check_index])
                        if dist < self.__sep:
                            return False
        return True

    def __isInRectangle(self, x, y):
        b_length = 1.0
        b_width = 0.5
        if x >= 0 and x <= b_length and y >= 0 and y <= b_width:
            return True
        else:
            return False

    def __isInCircle(self, x, y):
        b_radius = 0.5
        dist = math.sqrt((x-0.5)**2 + (y-0.5)**2)
        if dist <= b_radius:
            return True
        else:
            return False

    def draw(self):
        ax = plt.figure().add_subplot(111)
        ax.set_aspect('equal')
        for index in self.index_all:
            ax.plot(self.grid[index,0], self.grid[index,1], marker='.',markersize=2, color='k')

        plt.xlim(0, self.__size)
        plt.ylim(0, self.__size)
        plt.show()


if __name__ == "__main__":
    dots = Poisson2D(0.05)
    dots.generate()