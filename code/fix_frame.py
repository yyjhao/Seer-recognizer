# bg.py

# Performs background extraction on a video by averaging the
# pixels of every frame in the video

import cv2

cap = cv2.VideoCapture("./stitched.mpeg")
frame_width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

left = cv2.VideoCapture("./football_left.mp4")
fps = left.get(cv2.cv.CV_CAP_PROP_FPS)

fourcc = cv2.cv.CV_FOURCC(*"MPEG")
output = cv2.VideoWriter('./stitched_fixed.mpeg', fourcc, fps, (frame_width, frame_height))

count = 0
while True:
    count += 1
    print "frame", count
    ret, p = cap.read()
    if not ret:
        break
    output.write(p)

output.release()
