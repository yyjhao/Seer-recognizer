import numpy as np
import cv2
import util
from scipy import stats

# Read background image
BACKGROUND_IMG = cv2.imread('../images/stitched_background.png')

BLUE  = (255, 0, 0)
GREEN = (0, 255, 0)
RED   = (0, 0, 255)

# Linear scaled threshold value
# 0-1029 -> 13-52
def threshVal(row):
  return float(row) / 1029 * 38 + 13

# Linear scaled area filter
# 0-1029 -> 0-2300
def areaVal(row):
  return float(row) / 1029 * 2300

# Generates a mask of the field
def fieldMask(frame):
  ## Corners and center
  pts = np.zeros([4,2], dtype=np.int) 
  pts[0,:] = [189, 2556] # Top left
  pts[1,:] = [168, 4918] # Top right
  pts[2,:] = [960, 8119] # Bottom right (outside of the image)
  pts[3,:] = [972, 0] # Bottom left
  
  mask = util.quadrangleMask(pts, frame.shape)
  
  return mask

FIELD_MASK = fieldMask(BACKGROUND_IMG[:,:,0])

# Filter out contours that:
#   1. Are too small
#   2. Have a wide aspect ratio
#   3. Are outside the field boundary
def contourFilter(contour_bound):
  x, y, w, h = contour_bound[1]
  check = w * h > areaVal(y + h) \
          and float(w) / h < 1.2 \
          and FIELD_MASK[y + h, x + w / 2] != 0
  return check

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
  binary_frame = np.zeros(delta_E.shape, 'uint8')
  binary_frame[delta_E > thresh_vals] = 255

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

    # Calculate statistics for hue of all the pixels within the contour
    mean_hue = np.mean(hues)
    median_hue = np.median(hues)
    mode_hue = stats.mode(hues.flatten())[0][0]

    # Determine the player color based on hue statistics and position
    if max(median_hue, mode_hue, mean_hue) > 60:
      # blue player
      color = BLUE
    elif min(median_hue, mode_hue, mean_hue) < 30:
      # red player
      color = RED
    elif bounding_rect[0] > 5000:
      # blue goalie
      color = BLUE
    elif bounding_rect[0] < 3000:
      # red goalie
      color = RED
    else:
      # referee
      color = GREEN

    # Append the player bounding rectangle and color to the list
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
    cv2.imshow('Player detection', detection_frame)
    cv2.imwrite('../images/player_detection/detections/{}.png'.format(k), detection_frame)
    cv2.waitKey(0)

  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
