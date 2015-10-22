import cv2
import cv2.cv as cv
import numpy as np
import stitcher
import matplotlib.pyplot as plt

right_add = 3500
left_add = 3200

lsrc = np.zeros((4, 2), dtype="float32")
ldst = np.zeros((4, 2), dtype="float32")

lsrc[0] = [447, 660]
lsrc[1] = [1763, 314]
lsrc[2] = [1919, 333]
lsrc[3] = [1539, 1079]


ldst[0] = [3230 - 5000 + left_add, 1028]
ldst[1] = [5245 - 5000 + left_add, 243]
ldst[2] = [5408 - 5000 + left_add, 242]
ldst[3] = [5231 - 5000 + left_add, 1010]

lptran = cv2.getPerspectiveTransform(lsrc, ldst)

lsrc[0] = [0, 226]
lsrc[1] = [0, 1056]
lsrc[2] = [475, 924]
lsrc[3] = [407, 220]


ldst[0] = [6507 - 5000 + left_add, 234]
ldst[1] = [6435 - 5000 + left_add, 1003]
ldst[2] = [6920 - 5000 + left_add, 1002]
ldst[3] = [6920 - 5000 + left_add, 230]

rptran = cv2.getPerspectiveTransform(lsrc, ldst)

atran = np.float32([[1, 0, left_add], [0, 1, 0]])

def overlap(top_image, bottom_image):
    ret, tgray = cv2.threshold(cv2.cvtColor(top_image, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)
    ret, mtgray = cv2.threshold(cv2.cvtColor(bottom_image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    intersect = cv2.bitwise_and(tgray, mtgray)
    return bottom_image - cv2.bitwise_and(bottom_image, bottom_image, mask=intersect) + top_image

def stitch_pics(l, m, r):
    mt = cv2.warpAffine(m, atran, (m.shape[1] + left_add + right_add, m.shape[0]))
    lt = cv2.warpPerspective(l, lptran, (mt.shape[1], mt.shape[0]))
    rt = cv2.warpPerspective(r, rptran, (mt.shape[1], mt.shape[0]))
    return overlap(overlap(lt, mt), rt)

mid = cv2.VideoCapture("./football_mid.mp4")
left = cv2.VideoCapture("./football_left.mp4")
right = cv2.VideoCapture("./football_right.mp4")

left_w = left.get(cv.CV_CAP_PROP_FRAME_WIDTH)
mid_w = mid.get(cv.CV_CAP_PROP_FRAME_WIDTH)
right_w = right.get(cv.CV_CAP_PROP_FRAME_WIDTH)

h = left.get(cv.CV_CAP_PROP_FRAME_HEIGHT)

fps = left_w.get(cv2.cv.CV_CAP_PROP_FPS)

fourcc = cv2.cv.CV_FOURCC(*"MPEG")
output = cv2.VideoWriter('./stitched.mpeg', fourcc, fps, (int(left_add + mid_w + right_add), int(h)))

count = 0
while True:
    count += 1
    print "frame", count
    retl, l = left.read()
    retm, m = mid.read()
    retr, r = right.read()
    if not (retl and retm and retr):
        break
    output.write(stitch_pics(l, m, r))

output.release()
