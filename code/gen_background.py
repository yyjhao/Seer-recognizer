import cv2
import numpy as np

def genBg():
    # background = cv2.imread("../images/stitched_frames/0.png")
    cap = cv2.VideoCapture('../videos/stitched.mpeg')
    movie_shape = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)))
    background = np.zeros(movie_shape)
    cap.release()
    for i in range(0, 7200):
        frame = cv2.imread("../images/stitched_frames/{}.png".format(i))
        # _, frame = cap.read()
        background += frame
        print i
    
    background /= 7200
    cv2.imwrite("..images/stitched_background.png", background)
