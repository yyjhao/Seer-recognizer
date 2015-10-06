# bg.py

# Performs background extraction on a video by averaging the
# pixels of every frame in the video

import cv2
import numpy as np
import sys

def main():
  # Read video file from command line argument
  filename = sys.argv[1]
  cap = cv2.VideoCapture(filename)

  # Get and print video properties
  frame_width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
  frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

  print "Frame width:", frame_width
  print "Frame height:", frame_height
  print "FPS:", fps
  print "Frame count:", frame_count

  # Read video frames and perform averaging
  _, img = cap.read()
  avgImg = np.float32(img)
  for fr in range(1, frame_count):
    _, img = cap.read()
    avgImg = (img + fr * avgImg) / (fr + 1)

  # Convert into uint8 image
  normImg = cv2.convertScaleAbs(avgImg)

  # Save the new image
  tokens = filename.split('.')
  output_file = tokens[0] + '_background.jpg'
  cv2.imwrite(output_file, normImg)

  # Close video file
  cap.release()

if __name__ == "__main__":
  main()
