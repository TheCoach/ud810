import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('input/tsukuba-L.png',0)
imgR = cv.imread('input/tsukuba-R.png',0)
stereo = cv.StereoBM_create(numDisparities=16 * 1,  blockSize=15);
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()