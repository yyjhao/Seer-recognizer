import cv2
import cv2.cv as cv
import numpy as np

right_add = 3500
left_add = 3200

lsrc = np.zeros((4, 2), dtype="float32")
ldst = np.zeros((4, 2), dtype="float32")

lsrc[0] = [447, 660]
lsrc[1] = [1763, 314]
lsrc[2] = [1920, 333]
lsrc[3] = [1539, 1079]

ldst[0] = [3230 - 5000 + left_add + 230, 1028 - 30]
ldst[1] = [5245 - 5000 + left_add - 7, 243]
ldst[2] = [5408 - 5000 + left_add - 28, 242]
ldst[3] = [5231 - 5000 + left_add - 40, 1010]

lptran = cv2.getPerspectiveTransform(lsrc, ldst)

lsrc[0] = [0, 226]
lsrc[1] = [0, 1056]
lsrc[2] = [475, 924]
lsrc[3] = [407, 220]

ldst[0] = [6507 - 5000 + left_add + 15, 234]
ldst[1] = [6435 - 5000 + left_add, 1003]
ldst[2] = [6920 - 5000 + left_add + 5, 1002]
ldst[3] = [6920 - 5000 + left_add + 25, 230]

rptran = cv2.getPerspectiveTransform(lsrc, ldst)

atran = np.float32([[1, 0, left_add], [0, 1, 0]])
leftonetran = np.float32([[1, 0, -3], [0, 1, -1]])

def remove_left(img, left):
    tran = np.float32([[1, 0, -left], [0, 1, 0]])
    img = cv2.warpAffine(img, tran, img.shape)
    tran = np.float32([[1, 0, left], [0, 1, 0]])
    img = cv2.warpAffine(img, tran, img.shape)
    return img

def remove_right(img, right):
    tran = np.float32([[1, 0, right], [0, 1, 0]])
    img = cv2.warpAffine(img, tran, img.shape)
    tran = np.float32([[1, 0, -right], [0, 1, 0]])
    img = cv2.warpAffine(img, tran, img.shape)
    return img


left_cached = False
right_cached = False

left_intersect = None
right_intersect = None
left_intersect_line = None
right_intersect_line = None

def overlap(top_image, bottom_image, left):
    global left_cached
    global right_cached
    global left_intersect_line
    global right_intersect_line
    global left_intersect
    global right_intersect
    if left:
        if not left_cached:
            _, tgray = cv2.threshold(cv2.cvtColor(top_image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
            _, mtgray = cv2.threshold(cv2.cvtColor(bottom_image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
            left_intersect = cv2.bitwise_and(tgray, mtgray)
            shiftintersect = cv2.warpAffine(left_intersect, leftonetran, (left_intersect.shape[1], left_intersect.shape[0]))
            left_intersect_line = left_intersect - shiftintersect
            left_intersect_line = remove_left(left_intersect_line, left_add + 3)
            left_cached = True
        intersect = left_intersect
        intersect_line = left_intersect_line

        mtinter = cv2.bitwise_and(bottom_image, bottom_image, mask=intersect)
        linemtinter = cv2.bitwise_and(bottom_image, bottom_image, mask=intersect_line)
        linetinter = cv2.bitwise_and(top_image, top_image, mask=intersect_line)

        return bottom_image - mtinter + top_image + linemtinter - linetinter
    else:
        if not right_cached:
            _, tgray = cv2.threshold(cv2.cvtColor(top_image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
            _, mtgray = cv2.threshold(cv2.cvtColor(bottom_image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
            right_intersect = cv2.bitwise_and(tgray, mtgray)
            right_cached = True
        intersect = right_intersect

        mtinter = cv2.bitwise_and(bottom_image, bottom_image, mask=intersect)
        tinter = cv2.bitwise_and(top_image, top_image, mask=intersect)


        return bottom_image - mtinter + top_image - tinter + cv2.addWeighted(mtinter, 0.5, tinter, 0.5, 0)


def stitch_pics(l, m, r):
    mt = cv2.warpAffine(m, atran, (m.shape[1] + left_add + right_add, m.shape[0]))
    lt = cv2.warpPerspective(l, lptran, (mt.shape[1], mt.shape[0]))
    rt = cv2.warpPerspective(r, rptran, (mt.shape[1], mt.shape[0]))
    return overlap(overlap(lt, mt, True), rt, False)


l = cv2.imread("frame528l.jpg")
r = cv2.imread("frame528r.jpg")
m = cv2.imread("frame528m.jpg")

cv2.imwrite("s.jpg", stitch_pics(l, m, r))

mid = cv2.VideoCapture("./football_mid.mp4")
left = cv2.VideoCapture("./football_left.mp4")
right = cv2.VideoCapture("./football_right.mp4")

left_w = left.get(cv.CV_CAP_PROP_FRAME_WIDTH)
mid_w = mid.get(cv.CV_CAP_PROP_FRAME_WIDTH)
right_w = right.get(cv.CV_CAP_PROP_FRAME_WIDTH)

h = left.get(cv.CV_CAP_PROP_FRAME_HEIGHT)

fps = left.get(cv2.cv.CV_CAP_PROP_FPS)

fourcc = cv2.cv.CV_FOURCC(*"MPEG")

# count = 0
# while True:
#     count += 1
#     print "frame", count
#     retl, l = left.read()
#     retm, m = mid.read()
#     retr, r = right.read()
#     if not (retl and retm and retr):
#         break
#     if 525 < count < 560:
#         cv2.imwrite("frame" + str(count) + "l.jpg", l)
#         cv2.imwrite("frame" + str(count) + "r.jpg", r)
#         cv2.imwrite("frame" + str(count) + "m.jpg", m)

