import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('input/pair1-L.png',0)
imgR = cv.imread('input/pair1-R.png',0)
stereo = cv.StereoSGBM_create(numDisparities=16 * 6, blockSize=11);
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()