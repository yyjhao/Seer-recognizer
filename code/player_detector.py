import numpy as np
import cv2

# Read background image
BACKGROUND_IMG = cv2.imread('../images/stitched_background.png')
# BACKGROUND_IMG = cv2.imread('../images/background2.png')
BACKGROUND_IMG_HSV = np.float32(cv2.cvtColor(BACKGROUND_IMG, cv2.COLOR_BGR2HSV))
BACKGROUND_IMG_LAB = np.float32(cv2.cvtColor(BACKGROUND_IMG, cv2.COLOR_BGR2LAB))

class Color(object):
    BLUE  = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED   = (0, 0, 255)
    BLUE_GOALIE = (255, 50, 50)
    RED_GOALIE  = (50, 50, 255)

# Shadow removal parameters
VALUE_LOW_THRESH = 0.4
VALUE_HIGH_THRESH = 0.95
SATURATION_THRESH = 60
HUE_THRESH = 4

# Several linear scaled threshold values based on row
def threshVal(row):
    if row < 175:
        return 100
    if row < 200:
        return 40
    elif row < 470:
        # 0-1029 -> 11-55
        return float(row) / 1029 * 44 + 11
    elif row < 740:
        # 0-1029 -> 15-56
        return float(row) / 1029 * 41 + 15
    # 0-1029 -> 8-51
    return float(row) / 1029 * 43 + 8

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
def generateMask(pts):
    mask = np.zeros(BACKGROUND_IMG[:,:,0].shape)
    cv2.fillConvexPoly(mask, pts, 1)
    return mask

# Area of the field players are standing in
FIELD_PTS = np.array(
    [[2592, 199],
    [4892, 182],
    [5400, 288],
    [5948, 288],
    [5948, 408],
    [8500, 990],
    [-100, 990]])
FIELD_MASK = generateMask(FIELD_PTS)

# CUZ THE BLUE GOALIE IS A NINJA
BLUE_GOALIE_PTS = np.array(
    [[5240, 267],
    [5940, 267],
    [5940, 400],
    [5240, 400]])
BLUE_GOALIE_MASK = generateMask(BLUE_GOALIE_PTS).astype('bool')

# The field has darker shadows in this part of the field
SHADOW_PTS = np.array(
    [[3200, 190],
    [5030, 178],
    [9550, 1100],
    [-500, 1100],
    [1512, 504],
    [2576, 504]])
SHADOW_MASK = generateMask(SHADOW_PTS).astype('bool')

# Filter out contours that:
#   1. Are too small
#   2. Have a wide aspect ratio
#   3. Are outside the field boundary
def contourFilter(contour_bound):
    x, y, w, h = contour_bound[1]
    check = w * h > areaVal(y + h) \
                    and float(w) / h < 2.4 \
                    and FIELD_MASK[y + h, x + w / 2] != 0
    if BLUE_GOALIE_MASK[y + h, x + w / 2]:
        check = check and w * h > 1050
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
    mask = np.logical_and(mask, SHADOW_MASK)
    return mask

# Detects all players in the frame, returning a list of tuples containing the
# bounding rectangle and color of each player of the form ((x, y, w, h), color)
def getPlayers(frame):
    # Convert frame to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hue = frame_hsv[:,:,0]
    frame_value = frame_hsv[:,:,2]

    # Convert frame and background to Lab in order to calculate the Delta E
    # color difference according to the CIE76 formula
    # Reference: https://en.wikipedia.org/wiki/Color_difference
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab_diff_squared = np.square(np.float32(frame_lab) - BACKGROUND_IMG_LAB)
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
    thresh_vals[BLUE_GOALIE_MASK] = 10
    binary_frame = np.zeros(delta_E.shape, 'uint8')
    binary_frame[delta_E > thresh_vals] = 255
    # cv2.imwrite('../images/player_detection/binaries/{}.png'.format(k), binary_frame)
    binary_frame[shadowMask(frame_hsv)] = 0
    # cv2.imwrite('../images/player_detection/shadow_removed/{}.png'.format(k), binary_frame)

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
        for pt in contour:
            pt[0][0] = pt[0][0] - bounding_rect[0]
            pt[0][1] = pt[0][1] - bounding_rect[1]

        # Create a mask where the contour interiors are set to 255,
        # and all other values are 0. Use the mask to grab all hue
        # and values within the contour.
        mask = np.zeros([bounding_rect[3],bounding_rect[2]]).astype('uint8')
        cv2.drawContours(mask, [contour], 0, 255, -1)
        indices = np.nonzero(mask)
        shifted_indices = (indices[0]+bounding_rect[1],indices[1]+bounding_rect[0])
        hues = frame_hue[shifted_indices]
        shifted_hues = np.remainder(hues + 10, 180)
        values = frame_value[shifted_indices]

        # Calculate statistics for hue/value of all pixels within the contour
        mean_hue = np.mean(shifted_hues)
        mean_value = np.mean(values)

        # Determine the player color based on hue/value statistics
        if mean_hue > 65:
            # blue player
            color = Color.BLUE
        elif mean_hue > 53:
            if mean_value > 110:
                # goalies and referee
                color = Color.GREEN
            else:
                # blue player
                color = Color.BLUE
        elif mean_hue < 35:
            # red player
            color = Color.RED
        else:
            # goalies and referee
            color = Color.GREEN

        # Append the player bounding rectangle and color to the list
        if color != Color.GREEN or bounding_rect[1] < 800:
            player = [bounding_rect, color, mean_hue, mean_value]
            players.append(player)

    # Rightmost green player is blue goalie and
    # leftmost green player is red goalie
    getX = lambda player: player[0][0]
    green_players = [p for p in players if p[1] == Color.GREEN]
    sorted_green_players = sorted(green_players, key=getX)
    sorted_green_players[0][1] = Color.RED_GOALIE
    sorted_green_players[-1][1] = Color.BLUE_GOALIE

    # contours = [cb[0] for cb in contour_bounds]
    # contour_frame = np.zeros(frame.shape)
    # contour_frame[:,:,0] = delta_E
    # contour_frame[:,:,1] = delta_E
    # contour_frame[:,:,2] = delta_E
    # cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 1)
    # cv2.imwrite('../images/player_detection/contours/{}.png'.format(k), contour_frame)

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
        players = getPlayers(frame, k)
        detection_frame = frame.copy()
        for player in players:
            x, y, w, h = player[0]
            color = player[1]
            # Draw the bounding rectangle around each detected player
            cv2.rectangle(detection_frame, (x,y), (x+w,y+h), color, 1)
            text = "{}, {}, {}, {:.2f}, {}, {}".format(x, y, w * h, float(w) / h, int(player[2]), int(player[3]))
            cv2.putText(detection_frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255))

        # Show and save the player detected frame
        # cv2.imwrite('../images/player_detection/detections/{}.png'.format(k), detection_frame)
        print "frame", k

def main2():
    frame_count = 7200
    start_frame = 0
    end_frame = 650

    # Read each frame
    with open("players.txt", 'w') as fout:
        for k in range(start_frame, end_frame):
            frame = cv2.imread('../images/stitched_frames/{}.png'.format(k))
            # frame = cv2.imread('../images/test.png')

            # Detect players in frame
            players = getPlayers(frame)
            fout.write(str(players))
            fout.write("\n")

            # detection_frame = frame.copy()
            # for player in players:
            #   x, y, w, h = player[0]
            #   color = player[1]
            #   # Draw the bounding rectangle around each detected player
            #   cv2.rectangle(detection_frame, (x,y), (x+w,y+h), color, 1)

            # # Show and save the player detected frame
            # cv2.imwrite('../images/player_detection/detections/{}.png'.format(k), detection_frame)
            # quit()

            # Show and save the player detected frame
            print "frame", k

def main3():
    frame_count = 7200
    offset = 0

    with open("players_test6.txt", 'w') as fout:
        # Read each frame
        for k in range(frame_count-offset):
            frame = cv2.imread('../images/stitched_frames/{}.png'.format(k+offset))

            # Detect players in frame
            players = getPlayers(frame)
            # detection_frame = frame.copy()
            # for player in players:
            #     x, y, w, h = player[0]
            #     color = player[1]
            #     # Draw the bounding rectangle around each detected player
            #     cv2.rectangle(detection_frame, (x,y), (x+w,y+h), color, 1)
            #     text = "{}, {}, {}, {:.2f}, {}, {}".format(x, y, w * h, float(w) / h, int(player[2]), int(player[3]))
            #     cv2.putText(detection_frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255))

            # Show and save the player detected frame
            # cv2.imwrite('../images/player_detection/detections/{}.png'.format(k), detection_frame)
            p = [(pl[0], pl[1]) for pl in players]
            fout.write(str(p))
            fout.write("\n")
            print "frame", k+offset

if __name__ == "__main__":
    main3()
