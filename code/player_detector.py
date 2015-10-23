import numpy as np
import cv2

# Read background image
BACKGROUND_IMG = cv2.imread('../images/stitched_background.jpg')

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
  delta_E = delta_E.astype('uint8')

  # Threshold the Delta E difference so all values greater than the thresh_val
  # are set to 255 (foreground), and all values lower are set to 0 (background).
  # This creates a binary image used for contour detection.
  thresh_val = 28
  r, binary_frame = cv2.threshold(delta_E, thresh_val, 255, cv2.THRESH_BINARY)

  # Find all outer contours in the binary image
  contours, h = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

  # Filter out contours that are too small or have a wide aspect ratio.
  # The list also includes each contour's bounding rectangle.
  contour_bounds = [(c, cv2.boundingRect(c)) for c in contours if cv2.contourArea(c) > 100]
  contour_bounds = [cb for cb in contour_bounds if float(cb[1][2]) / cb[1][3] < 1]

  players = []

  for cb in contour_bounds:
    contour = cb[0]
    bounding_rect = cb[1]

    # Create a mask where the contour interiors are set to 255,
    # and all other values are 0
    mask = np.zeros(binary_frame.shape).astype('uint8')
    cv2.drawContours(mask, [contour], 0, 255, -1)

    # Calculate the average hue of all the pixels within the contour
    avg_hue = cv2.mean(frame_hue, mask=mask)[0]

    # Determine the player color based on the average hue
    if avg_hue > 60:
      color = (255, 0, 0)
    elif avg_hue > 30:
      color = (0, 255, 0)
    else:
      color = (0, 0, 255)

    # Append the player bounding rectangle and color to the list
    player = (bounding_rect, color)
    players.append(player)

  return players

# Runs player detection on each frame of the stitched football video.
# Each player detected frame is then saved upon stepping through the frames.
def main():
  # Open video file
  cap = cv2.VideoCapture('../videos/stitched_fixed.mpeg')
  frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

  # Read each frame
  for k in range(frame_count):
    _, frame = cap.read()

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
    cv2.imwrite('../images/player_detection/frame_{}.jpg'.format(k), detection_frame)
    cv2.waitKey(0)

  cv2.destroyAllWindows()
  cap.release()

if __name__ == "__main__":
  main()
