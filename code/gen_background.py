import cv2
import cv2.cv as cv
import numpy as np
import stitcher
import matplotlib.pyplot as plt
import sys

background = cv2.imread("../images/stitched_frames/0.png")
background = np.zeros(background.shape)
for i in range(0, 7200):
    frame = cv2.imread("../images/stitched_frames/{0}.png".format(i))
    background += frame
    print i

background /= 7200
cv2.imwrite("background2.png", background)