'''

    Generates a video based on player.txt
    
'''

import cv2
import numpy as np
import copy
import heatmap
from player_tracker import PlayerTracker
from player_detector import Color

PATH_SMOOTHED_TOP_DOWN_DATA = "players_smoothedTopDown.txt"

BORDER = 100
PATH_TOP_DOWN_IMG = '../images/FootballField_small_border.png'
WIDTH_TD_IMG = 1400
HEIGHT_TD_IMG = 1000

# Dimensions of the field in m without borders
FIELD_HEIGHT = 70.0
FIELD_WIDTH = 105.0

# Distance of one pixel
PX_TO_M = FIELD_WIDTH / (WIDTH_TD_IMG - 2 * BORDER) 

'''
    Calculate the homography matrix to map the orignal field to the 
    top down field using 21 manually selected points.
    
    @return 3x3 Homography matrix
'''
def getHomographyMatrix():
    img = cv2.imread(PATH_TOP_DOWN_IMG)
    
    # ========= The 21 points on the top down view ========= #    
    target_pts = np.zeros([21, 2])
    # y,x (row, col)
    target_pts[0, :] = [BORDER + 295, BORDER + 69]
    target_pts[1, :] = [BORDER + 505, BORDER + 69]
    target_pts[2, :] = [BORDER + 170, BORDER + 190]
    target_pts[3, :] = [BORDER + 630, BORDER + 190]
    target_pts[4, :] = [BORDER + 400, BORDER + 230]
    target_pts[5, :] = [BORDER + 400, BORDER + 494]
    target_pts[6, :] = [BORDER + 400, BORDER + 706]
    target_pts[7, :] = [BORDER + 400, BORDER + 970]
    target_pts[8, :] = [BORDER + 170, BORDER + 1010]
    target_pts[9, :] = [BORDER + 630, BORDER + 1010]
    target_pts[10, :] = [BORDER + 295, BORDER + 1131]
    target_pts[11, :] = [BORDER + 505, BORDER + 1131]
    target_pts[12, :] = [BORDER + 295, BORDER + 1195]
    target_pts[13, :] = [BORDER + 505, BORDER + 1195]
    target_pts[14, :] = [BORDER + 295, BORDER + 5]
    target_pts[15, :] = [BORDER + 505, BORDER + 5]
    # # Corners and center
    target_pts[16, :] = [BORDER, BORDER]  # Top left (image is orientated to the top left)
    target_pts[17, :] = [BORDER, img.shape[1] - 1 - BORDER]  # Top right
    target_pts[18, :] = [ img.shape[0] - 1 - BORDER, img.shape[1] - 1 - BORDER]  # Bottom right
    target_pts[19, :] = [img.shape[0] - 1 - BORDER, BORDER]  # Bottom left
    target_pts[20, :] = [(img.shape[0]) / 2, img.shape[1] / 2]  # Center points
    
    # ========= The 21 points on the normal field  ========= # 
    # Point format (row, col) == (y,x)
    pts = np.zeros([21, 2])
    # # Special points on the image
    pts[0, :] = [300, 2370]  # A (in the white)
    pts[1, :] = [443, 1929]  # B (in the white)
    pts[2, :] = [245, 2817]  # C (in the white)
    pts[3, :] = [591, 2000]  # D (in the white)
    pts[4, :] = [358, 2664]  # E (in the white)
    pts[5, :] = [359, 3454]  # F (in the white)
    pts[6, :] = [355, 4083]  # G (in the white)
    pts[7, :] = [340, 4880]  # H (in the white)
    pts[8, :] = [231, 4678]  # I (in the white)
    pts[9, :] = [576, 5705]  # J (in the white)
    pts[10, :] = [280, 5176]  # K (in the white)
    pts[11, :] = [421, 5722]  # L (in the white)
    pts[12, :] = [279, 5341]  # M (in the white)
    pts[13, :] = [420, 5948]  # N (in the white)
    pts[14, :] = [301, 2219]  # O (in the white)
    pts[15, :] = [445, 1721]  # P (in the white)
    # # Corners and center
    pts[16, :] = [196, 2593]  # Top left (outer coord)
    pts[17, :] = [177, 4892]  # Top right (outer coord)
    pts[18, :] = [950, 8206]  # Bottom right (outer coord)
    pts[19, :] = [942, 40]  # Bottom left (outer coord)
    pts[20, :] = [350, 3767]  # Center
    
    # Calculate the homography matrix, which will be used to project any point from the video to the top down view.
    return homography(target_pts, pts) 

'''
    Uses a homography matrix to transform a 2d point on the source image
    to a 2d point on the target image

    @param H: Homography matrix
    @param point: A tuple containing the row (y) and col (x) value of the to mapping point
    
    @retun A tuple containing the row (y) and col (x) value of the new point.
'''
def getTransformationCoords(H, point):
    # Transform
    tmp = np.dot(H, np.array([point[0], point[1], 1]))
    # Normalize
    tmp = tmp / tmp[2]
    return [int(tmp[0]), int(tmp[1])]
    
'''
    Uses a inverted homography matrix to calculate 
    from the target image to the source image
    
    @param Hinv: inverted homography matrix
    @param row: The row (y) value of the to mapping point
    @param col: The col (x) value of the to mapping point
    
    @retun A tuple containing the row (y) and col (x) value of the new point.
'''
def getInverseTransformationCoords(Hinv, row, col):
    tmp = np.dot(Hinv, np.array([row, col, 1]))
    tmp = tmp / tmp[2];
    return (int(tmp[0]), int(tmp[1]))

'''
    Calculates the homograpgy matrix given source and target points.
    
    @param target_pts: The point on the image where the matrix should map points to.
    @param source_pts: The point on the original image
    
    @return 3x3 Homography matrix
'''
def homography(target_pts, source_pts):
    A = np.zeros([target_pts.shape[0] * 2, 9])
    
    # Fill A
    for i in xrange(target_pts.shape[0]):
        A[2 * i, :] = [source_pts[i, 0], source_pts[i, 1], 1, 0, 0, 0, -target_pts[i, 0] * source_pts[i, 0], -target_pts[i, 0] * source_pts[i, 1], -target_pts[i, 0]]
        A[2 * i + 1, :] = [0, 0, 0, source_pts[i, 0], source_pts[i, 1], 1, -target_pts[i, 1] * source_pts[i, 0], -target_pts[i, 1] * source_pts[i, 1], -target_pts[i, 1]]
    
    # Solve A and extract H
    _, _, V = np.linalg.svd(A)  # U,S,V
    H = V[8].reshape((3, 3))    

    # Normalize H and return the matrix
    return H / H[2][2]

'''
    Creates the top down video using the player data 
    from the smoothed data file
'''
def createTopDownVideo():
    with open(PATH_SMOOTHED_TOP_DOWN_DATA) as fin:
        players_list = [eval(line) for line in fin]
        
    img = cv2.imread(PATH_TOP_DOWN_IMG)
    
    fps = 23
    movie_shape = (img.shape[1], img.shape[0])

    fourcc = cv2.cv.CV_FOURCC(*"MPEG")
    output = cv2.VideoWriter('./topDown.mpeg', fourcc, fps, movie_shape)
    
    for i, players in enumerate(players_list):
        newImg = copy.copy(img)
        print "frame", i
        
        _, newImg = addPlayers(newImg, players, str(i))
        output.write(newImg)
        
    output.release()

def detectedPlayers(players_list, H):
    counters = {}
    tracker = PlayerTracker(players_list[0])
    for p in players_list[1:]:
        tracker.feed_rectangles(p)
    r = [[] for i in players_list]
    player_points = [
        smooth([bound_point(project_point(point, H)) for point in p.raw_positions])
        for p in tracker.players
    ]
    for p, points in zip(tracker.players, player_points):
        num = counters.get(p.color, 1)
        counters[p.color] = num + 1
        for i, point in enumerate(points):
            r[i].append((point, p.color, str(num)))
    return r

'''
    Same as getTransformationCorrds, just for points where y and x are switched.
'''
def project_point(point, H):
    row = point[1]
    col = point[0]
    a = getTransformationCoords(H, [row, col])
    return (a[1], a[0])

def bound_point(point):
    return (point[0], max(point[1], BORDER))


def smooth_num(num_list):
    weight = 0.8
    for _ in range(50):
        pre = None
        nxt = None
        new_list = []
        for i, p in enumerate(num_list):
            val = p * weight
            if i + 1 < len(num_list):
                nxt = num_list[i + 1]
            else:
                nxt = None
            if not pre:
                val += nxt * (1 - weight)
            elif not nxt:
                val += pre * (1 - weight)
            else:
                val += nxt * (1 - weight) / 2 + pre * (1 - weight) / 2
            new_list.append(val)
            pre = p
        num_list = new_list
    return [int(round(i)) for i in num_list]

def smooth(point_list):
    return zip(*[smooth_num(l) for l in zip(*point_list)])

'''
    Draw all players from the given player list on the top down field.
    
    @param img: Image to draw the players on
    @param players: The list of the players with their position
    @param frameInd: ID of the frame, used for debugging
    
    @return A tuple, the first part contains the list of all player which have been drawn 
                successfully on the map, the second part contains the image with the players drawn on
'''
font = cv2.FONT_HERSHEY_SIMPLEX
def addPlayers(img, players, frameInd):
    a = np.zeros([2])
    playersOnTheField = []
    for ind, player in enumerate(players):
        a = (player[0][1], player[0][0])
        try:
            for i in xrange(5):
                for j in xrange(5):
                    if i + j <= 5 :
                        img[int(a[0] + i), int(a[1] + j)] = player[1]
                        img[int(a[0] + i), int(a[1] - j)] = player[1]
                        img[int(a[0] - i), int(a[1] + j)] = player[1]
                        img[int(a[0] - i), int(a[1] - j)] = player[1]
            # If we reach that point the player was somewhere on the field
            playersOnTheField.append((player, (a[0], a[1])))
        except IndexError:
            print 'Player ' + str(ind) + ' out side the field!'

        name = player[2] if player[1][1] > 0 else str(int(player[2]) + 1)
        cv2.putText(img, player[2], (a[1] - 30, a[0] - 10), font, 1, player[1], 2)

        # cv2.putText(img, "Frame: " + frameInd, (100, 100), font, 1, (255, 255, 255), 2)

    return playersOnTheField, img

'''
    Function to calculate the distance in meters between two top down coordinates

    Parameters:
        pos1: position 1, np.array([x,y])
        pos2: position 2, np.array([x,y])
'''
def distanceBetweenCoordsTopDown(pos1, pos2):
    # Calculate distance between two vectors in PIXEL
    distance = np.sqrt(np.sum(np.square(pos1 - pos2)))
        
    return distance * PX_TO_M

'''
    Evaluates the homography mapping by transforming 21 points from the original image 
    to the top down view and draw them on the field and write this image to the disk.
    Therewith, the user can easily see the precision of the mapping.
'''
def evalMapping():
    img = cv2.imread(PATH_TOP_DOWN_IMG)
    H = getHomographyMatrix()
    
    # # Points on the image (row, col) == (y,x)
    pts = np.zeros([21, 2])
    # # Special points on the image
    pts[0, :] = [300, 2370]  # A (in the white)
    pts[1, :] = [443, 1929]  # B (in the white)
    pts[2, :] = [245, 2817]  # C (in the white)
    pts[3, :] = [591, 2000]  # D (in the white)
    pts[4, :] = [358, 2664]  # E (in the white)
    pts[5, :] = [359, 3454]  # F (in the white)
    pts[6, :] = [355, 4083]  # G (in the white)
    pts[7, :] = [340, 4880]  # H (in the white)
    pts[8, :] = [231, 4678]  # I (in the white)
    pts[9, :] = [576, 5705]  # J (in the white)
    pts[10, :] = [280, 5176]  # K (in the white)
    pts[11, :] = [421, 5722]  # L (in the white)
    pts[12, :] = [279, 5341]  # M (in the white)
    pts[13, :] = [420, 5948]  # N (in the white)
    pts[14, :] = [301, 2219]  # O (in the white)
    pts[15, :] = [445, 1721]  # P (in the white)
        
    # # Corners and center
    pts[16, :] = [196, 2593]  # Top left (outer coord)
    pts[17, :] = [177, 4892]  # Top right (outer coord)
    pts[18, :] = [950, 8206]  # Bottom right (outer coord)
    pts[19, :] = [942, 40]  # Bottom left (outer coord)
    pts[20, :] = [350, 3767]  # Center
    
    a = np.zeros([2])
    i = 0
    for pt in pts:
        i = i + 1
        a = getTransformationCoords(H, [pt[0], pt[1]])
        try:
            for i in xrange(5):
                for j in xrange(5):
                    if i + j <= 5 :
                        img[int(a[0] + i), int(a[1] + j)] = [0, 0, 255]
                        img[int(a[0] + i), int(a[1] - j)] = [0, 0, 255]
                        img[int(a[0] - i), int(a[1] + j)] = [0, 0, 255]
                        img[int(a[0] - i), int(a[1] - j)] = [0, 0, 255]
        except IndexError:
            print 'Player ' + str(i) + ' out side the field!'
    
    cv2.imwrite('EvalField1.jpg', img)        

'''
    Generates the smoothed top down player positions and writes them in a globally defined file.
    
    @param inputFilePath: the path to the file containing the player positions.
'''
def playerDataToSmoothedTopDown(inputFilePath):
    with open(inputFilePath) as fin:
        players_list = [ eval(line) for line in fin ]

        for i in range(len(players_list)):
            players_list[i] = [p for p in players_list[i] if keepRect(p[0])]
        H = getHomographyMatrix()
        smoothedData = detectedPlayers(players_list, H)

    with open(PATH_SMOOTHED_TOP_DOWN_DATA, 'w') as fout:
        for frameData in smoothedData:
            fout.write(str(frameData))
            fout.write("\n")
'''
    Evaluation method to check wheater a rectangle should be kept or not (depending on the size)
    
    @param rect: The rectangle which should be checked (x,y,w,h)
'''
def keepRect(rect):
    if rect[1] + rect[3] > 950:
        print "removed a rect"
        return False
    return True

'''
    Convert the datastructure of the player which contains the positions and player 
    ids for each frame to another data structure, providing for each team, for each 
    player all his locations in a list, such that a heatmap or the walked distance 
    can be created easily.
    
    @return Tuple with the framewise and playerwise data
'''
def getPlayerWiseTopDown():
    with open(PATH_SMOOTHED_TOP_DOWN_DATA) as fin:
        framewiseData = [eval(line) for line in fin]
        
        # Create dict matching players to array id
        
        # Playerwise Data
        # Team (Referee, Red, Blue), Playerpositions
        playersPerTeam = np.zeros(2)
        for player in framewiseData[0]:  # # Use frame 5
            if player[1] == Color.RED or player[1] == Color.RED_GOALIE:
                playersPerTeam[0] += 1
            elif player[1] == Color.BLUE or player[1] == Color.BLUE_GOALIE:
                playersPerTeam[1] += 1
                    
        playerwisePositions = np.zeros((playersPerTeam[0], len(framewiseData), 2)), np.zeros((playersPerTeam[1], len(framewiseData), 2))
        # 0 = referee
        # 1 = red
        # 2 = blue
        for i, frame in enumerate(framewiseData):
            # Sample player data: ((852, 665), (0, 0, 255), '1')
            # 0: position
            # 1: color
            # 2: id for color
            for player in frame:
                if player[1] == Color.RED:
                    playerwisePositions[0][int(player[2]), i, :] = player[0][::-1]
                elif player[1] == Color.RED_GOALIE:
                    playerwisePositions[0][int(player[2]) - 1, i, :] = player[0][::-1]
                elif player[1] == Color.BLUE: 
                    playerwisePositions[1][int(player[2]), i, :] = player[0][::-1]
                elif player[1] == Color.BLUE_GOALIE:
                    playerwisePositions[1][int(player[2]) - 1, i, :] = player[0][::-1]
        
    return framewiseData, playerwisePositions

'''
    Generates the distances each player walked
    
    @param The data structure containing for each player by team all his positions.
    
    @return The distances walked after each frame for each player by team
'''
def getDistancesWalkedFramewise(playerPos):
    playerDist = np.zeros((playerPos[0].shape[0], playerPos[0].shape[1])), np.zeros((playerPos[1].shape[0], playerPos[1].shape[1]))
    for t, team in enumerate(playerPos):
        for p, player in enumerate(team):
            for i in xrange(1, player.shape[0]):
                dist = distanceBetweenCoordsTopDown(player[i - 1], player[i])
                if dist > 1: dist = 1  # Remove outliers
                playerDist[t][p, i] = dist + playerDist[t][p, i - 1]
                
    return playerDist

'''
    Generates a heatmap for each player and stores it on the disk.
    
    @param The data structure containing for each player by team all his positions.
'''
def generateHeatmaps(playerPos):
    for t, team in enumerate(playerPos):
        for p, player in enumerate(team):
            print "Generate Heatmaps for team " + str(t) + " player " + str(p)
            cv2.imwrite("../images/heatmaps/team" + str(t) + "_player" + str(p) + ".png", heatmap.getFieldHeatmap(player))


'''
    Draws a offside line on the normal view, depending on the positions of the players in the top down view.
    If a player of the attacking team is closer to the goal than any defending player except the goaly,
    then it will draw a line adjusted on the last defending player.
    There can be no line, one line or even two lines.
    
    @param players: The framewise player data, with (x,y) coordinates
    @param img: The image of the normal view to draw the line on
    @param Hinv: The inverted homography matrix.
    
    @return the image of the field with the offside line if the they are necessary.
'''
def drawOffsetLines(players, img, Hinv):
    # blue is on the right side
    # red is on the left side (defined by user)
    # Colors of players on the left and right side - goalies should have a slightly other color!
    left = Color.RED
    right = Color.BLUE
    teamLeft_mostLeft = 9999999
    teamLeft_mostRight = -1
    teamRight_mostLeft = 9999999
    teamRight_mostRight = -1
    
    for player in players:
        if player[1] == left:
            if player[0][0] < teamLeft_mostLeft:
                teamLeft_mostLeft = player[0][0]
            if player[0][0] > teamLeft_mostRight:
                teamLeft_mostRight = player[0][0]
        elif player[1] == right:
            if player[0][0] < teamRight_mostLeft:
                teamRight_mostLeft = player[0][0]
            if player[0][0] > teamRight_mostRight:
                teamRight_mostRight = player[0][0]
            
            
    imgLeftLine = None
    imgRightLine = None
    
    # Left = small number
    if teamLeft_mostLeft > teamRight_mostLeft and teamRight_mostLeft < WIDTH_TD_IMG / 2:
        # Offside on the left
        # Get coords of the line
        # Hinv = np.linalg.inv(H)
        topPt = getInverseTransformationCoords(Hinv, BORDER, teamLeft_mostLeft)
        bottomPt = getInverseTransformationCoords(Hinv, img.shape[0] - BORDER, teamLeft_mostLeft)
        
        # Blend the new line, such that it is transparent
        imgLeftLine = copy.copy(img) 
        cv2.line(imgLeftLine, (topPt[1], topPt[0]), (bottomPt[1], bottomPt[0]), (0, 0, 255), 10)
        
        
    # If there is a player from the left team more right than any player from team right
    # and this player is on the right players half (right = large number)  
    if teamLeft_mostRight > teamRight_mostRight and teamLeft_mostRight > WIDTH_TD_IMG / 2:
        # Offside on the right
        # Get coords of the line
        # Hinv = np.linalg.inv(H)
        topPt = getInverseTransformationCoords(Hinv, BORDER, teamRight_mostRight)
        bottomPt = getInverseTransformationCoords(Hinv, img.shape[0] - BORDER, teamRight_mostRight)

        # Blend the new line, such that it is transparent
        imgRightLine = copy.copy(img) 
        cv2.line(imgRightLine, (topPt[1], topPt[0]), (bottomPt[1], bottomPt[0]), (0, 0, 255), 10)
        
    # Most likely therefore the first one 
    if imgRightLine is None and imgLeftLine is None:
        return img
    
    if imgRightLine is not None and imgLeftLine is not None:
        img = cv2.addWeighted(img, 0.33333, imgRightLine, 0.66667, 0)
        return cv2.addWeighted(img, 0.6, imgLeftLine, 0.4, 0)
    
    if imgRightLine is not None:
        return cv2.addWeighted(img, 0.6, imgRightLine, 0.4, 0)
    
    return cv2.addWeighted(img, 0.6, imgLeftLine, 0.4, 0)
    

if __name__ == '__main__':
    # evalMapping()
    # playerDataToSmoothedTopDown("players_cat.txt")
    # _, playersPos = getPlayerWiseTopDown()
    # getDistancesWalkedFramewise(playerPos)
    # generateHeatmaps(playersPos)
    createTopDownVideo()
