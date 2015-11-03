import numpy as np
import cv2
import util
from scipy import stats

# Read background image
BACKGROUND_IMG = cv2.imread('../images/stitched_background.png')
BACKGROUND_IMG_HSV = np.float32(cv2.cvtColor(BACKGROUND_IMG, cv2.COLOR_BGR2HSV))

class Color(object):
  BLUE  = (255, 0, 0)
  GREEN = (0, 255, 0)
  RED   = (0, 0, 255)
  BLUE_GOALIE = (255, 50, 50)
  RED_GOALIE  = (50, 50, 255)

# Shadow removal parameters
VALUE_LOW_THRESH = 0.65
VALUE_HIGH_THRESH = 1
SATURATION_THRESH = 50
HUE_THRESH = 5

# Several linear scaled threshold values based on row
def threshVal(row):
  if row < 190:
    return 50
  elif row < 470:
    # 0-1029 -> 9-53
    return float(row) / 1029 * 44 + 9
  elif row < 740:
    # 0-1029 -> 18-59
    return float(row) / 1029 * 41 + 18
  # 0-1029 -> 11-54
  return float(row) / 1029 * 43 + 11

# Several linear scaled area filters based on row
def areaVal(row):
  if row < 260:
    # 0-1029 -> 0-1200
    return float(row) / 1029 * 1200
  if row < 470:
    # 0-1029 -> 0-1700
    return float(row) / 1029 * 1700
  elif row < 740:
    # 0-1029 -> 0-2200
    return float(row) / 1029 * 2200
  # 0-1029 -> 0-4300
  return float(row) / 1029 * 4300

# Generates a mask where the area inside the polygon
# specified by the 4 pts is set to 1, and everything
# else is set to 0
def generateMask(pt0, pt1, pt2, pt3):
  pts = np.zeros([4,2], dtype=np.int) 
  pts[0,:] = pt0 # Top left
  pts[1,:] = pt1 # Top right
  pts[2,:] = pt2 # Bottom right
  pts[3,:] = pt3 # Bottom left
  
  mask = util.quadrangleMask(pts, BACKGROUND_IMG[:,:,0].shape)
  
  return mask

# Area of the field players are standing in
FIELD_MASK = generateMask(
  [190, 2520],
  [166, 4990],
  [990, 8700],
  [1000, -470])

# CUZ THE BLUE GOALIE IS A NINJA
BLUE_GOALIE_MASK = generateMask(
  [267, 5364],
  [267, 5700],
  [400, 5700],
  [400, 5364]).astype('bool')

# Filter out contours that:
#   1. Are too small
#   2. Have a wide aspect ratio
#   3. Are outside the field boundary
def contourFilter(contour_bound):
  x, y, w, h = contour_bound[1]
  check = w * h > areaVal(y + h) \
          and float(w) / h < 2.4 \
          and FIELD_MASK[y + h, x + w / 2] != 0
  return check

# Creates a mask where all shadows identified in the given frame
# are given the value True, and all other points are False
def shadowMask(frame_hsv):
  frame_hsv_float = np.float32(frame_hsv)
  hue_diff = np.absolute(frame_hsv_float[:,:,0] - BACKGROUND_IMG_HSV[:,:,0])
  hue = np.minimum(hue_diff, 360 - hue_diff)
  saturation = np.absolute(frame_hsv_float[:,:,1] - BACKGROUND_IMG_HSV[:,:,1])
  value = frame_hsv_float[:,:,2] / BACKGROUND_IMG_HSV[:,:,2]

  mask = np.logical_and(VALUE_LOW_THRESH <= value, value <= VALUE_HIGH_THRESH)
  mask = np.logical_and(saturation <= SATURATION_THRESH, mask)
  mask = np.logical_and(hue <= HUE_THRESH, mask)
  return mask

# Detects all players in the frame, returning a list of tuples containing the
# bounding rectangle and color of each player of the form ((x, y, w, h), color)
def getPlayers(frame):
  # Convert frame to HSV
  frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  frame_hue = frame_hsv[:,:,0]

  # Convert frame and background to Lab in order to calculate the Delta E
  # color difference according to the CIE76 formula
  # Reference: https://en.wikipedia.org/wiki/Color_difference
  frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
  bg_lab = cv2.cvtColor(BACKGROUND_IMG, cv2.COLOR_BGR2LAB)
  lab_diff_squared = np.square(np.float32(frame_lab) - np.float32(bg_lab))
  delta_E = np.sqrt(np.sum(lab_diff_squared, axis=2))

  # Normalize the Delta E difference to a value between 0-255
  delta_E = delta_E / delta_E.max() * 255

  # Threshold the Delta E difference so all values greater than the thresh_val
  # are set to 255 (foreground), and all values lower are set to 0 (background).
  # This creates a binary image used for contour detection.
  # Use variable thresholding based on the row coordinate of the image.
  thresh_vals = np.zeros(delta_E.shape)
  for row in range(thresh_vals.shape[0]):
    thresh_vals[row,:] = threshVal(row)
  thresh_vals[BLUE_GOALIE_MASK] = 8
  binary_frame = np.zeros(delta_E.shape, 'uint8')
  binary_frame[delta_E > thresh_vals] = 255
  binary_frame[shadowMask(frame_hsv)] = 0

  # Find all outer contours in the binary image
  contours, h = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

  # Filter out contours that are not players on the field
  contour_bounds = [(c, cv2.boundingRect(c)) for c in contours]
  contour_bounds = filter(contourFilter, contour_bounds)

  players = []

  for cb in contour_bounds:
    contour = cb[0]
    bounding_rect = cb[1]

    # Create a mask where the contour interiors are set to 255,
    # and all other values are 0. Use the mask to grab all hue
    # values within the contour.
    mask = np.zeros(binary_frame.shape).astype('uint8')
    cv2.drawContours(mask, [contour], 0, 255, -1)
    hues = frame_hue[np.nonzero(mask)]
    shifted_hues = np.remainder(hues + 10, 180)

    # Calculate statistics for hue of all the pixels within the contour
    mean_hue = np.mean(shifted_hues)

    # Determine the player color based on hue statistics and position
    if mean_hue > 60:
      # blue player
      color = Color.BLUE
    elif mean_hue < 35:
      # red player
      color = Color.RED
    elif bounding_rect[0] > 5000 and bounding_rect[1] < 800:
      # blue goalie
      color = Color.BLUE_GOALIE
    elif bounding_rect[0] < 3000:
      # red goalie
      color = Color.RED_GOALIE
    else:
      # referee
      color = Color.GREEN

    # Append the player bounding rectangle and color to the list
    if color != Color.GREEN or bounding_rect[1] < 800:
      player = (bounding_rect, color)
      players.append(player)

  return players

# Runs player detection on each frame of the stitched football video.
# Each player detected frame is then saved upon stepping through the frames.
# NOTE: Frames are read from a folder of PNG files instead of the MPEG
# video in order to combat lossy video compression
def main():
  frame_count = 7200

  # Read each frame
  for k in range(frame_count):
    frame = cv2.imread('../images/stitched_frames/{}.png'.format(k))

    # Detect players in frame
    players = getPlayers(frame)
    detection_frame = frame.copy()
    for player in players:
      x, y, w, h = player[0]
      color = player[1]
      # Draw the bounding rectangle around each detected player
      cv2.rectangle(detection_frame, (x,y), (x+w,y+h), color, 1)

    # Show and save the player detected frame
    cv2.imwrite('../images/player_detection/detections/{}.png'.format(k), detection_frame)
    print "frame", k
    cv2.imshow("Player detection", detection_frame)
    cv2.waitKey(0)

  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
