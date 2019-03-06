import numpy as np
import cv2 as cv
import math

def get_template_position(path):
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            if (idx == 0):
                x = int(float(line.split()[0]))
                y = int(float(line.split()[1]))
            if (idx == 1):
                w = int(float(line.split()[0]))
                if (w % 2 == 0):
                    w += 1
                h = int(float(line.split()[1]))
                if (h % 2 == 0):
                    h += 1

    return x, y, w, h
