import numpy as np
import cv2

def main():
  cap = cv2.VideoCapture('../videos/football_mid.mp4')
  while True:
    ret, frame = cap.read()
    if ret:   
      # convert frame to grayscale
      frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # insert frame processing code here

    else:
      break
  cap.release()

if __name__ == "__main__":
  print "men playing football yes football"
  main()
