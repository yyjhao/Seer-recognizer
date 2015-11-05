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

def getHomographyMatrix():
    img = cv2.imread(PATH_TOP_DOWN_IMG)  # , cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    # Define the 4 corner of the football field (target points), thereby the width will be kept and only the height adjusted
    # so we don't accidently lose to many information
    # [y,x]
    
    
    # newImg = np.zeros([ratio*img.shape[1], img.shape[1],3])
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
    
    
    
    # Calculate the homography matrix, which will be used to project any point from the video to the top down view.
    return homography(target_pts, pts) 
    
# Transforms 2d points to 2d points on another plane
def getTransformationCoords(H, point):
    # Transform
    tmp = np.dot(H, np.array([point[0], point[1], 1]))
    # Normalize
    tmp = tmp / tmp[2]
    return [int(tmp[0]), int(tmp[1])]
    
def getInverseTransformationCoords(Hinv, row, col):
    tmp = np.dot(Hinv, np.array([row, col, 1]))
    tmp = tmp / tmp[2];
    return (int(tmp[0]), int(tmp[1]))
        
# I assume projPts=u_p,v_p, pts=u_c,v_c
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

def createTopDownVideo(players_list):
    img = cv2.imread(PATH_TOP_DOWN_IMG)  # , cv2.CV_LOAD_IMAGE_GRAYSCALE)
    H = getHomographyMatrix()
    
    cap = cv2.VideoCapture('../videos/stitched.mpeg')

    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    movie_shape = (img.shape[1], img.shape[0])

    fourcc = cv2.cv.CV_FOURCC(*"MPEG")
    output = cv2.VideoWriter('./topDown1.mpeg', fourcc, fps, movie_shape)
    
    for i, players in enumerate(players_list):
        newImg = copy.copy(img)
        print "frame", i
        
        _, newImg = addPlayers(newImg, players, str(i))
        output.write(newImg)
        
    cap.release()
    output.release()

def detectedPlayers(players_list, H):
    counters = {}
    tracker = PlayerTracker(players_list[0])
    for p in players_list[1:]:
        tracker.feed_rectangles(p)
    r = [[] for i in players_list]
    player_points = [
        smooth([project_point(point, H) for point in p.raw_positions])
        for p in tracker.players
    ]
    for p, points in zip(tracker.players, player_points):
        num = counters.get(p.color, 1)
        counters[p.color] = num + 1
        for i, point in enumerate(points):
            r[i].append((point, p.color, str(num)))
    return r

def project_point(point, H):
    row = point[1]
    col = point[0]
    a = getTransformationCoords(H, [row, col])
    return (a[1], a[0])


def smooth_num(num_list):
    weight = 0.9
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

    
font = cv2.FONT_HERSHEY_SIMPLEX
# Add all players in the given players list to the field
# img: image to add the 
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
        cv2.putText(img, player[2], (a[1] - 30, a[0] - 10), font, 1, player[1], 2)
        cv2.putText(img, "Frame: " + frameInd, (100, 100), font, 1, (255, 255, 255), 2)

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
    # cv2.imshow('new img', img)
    cv2.waitKey(0)

def playerDataToSmoothedTopDown(inputFilePath, outputFilePath):
    with open(inputFilePath) as fin:
        players_list = [ eval(line) for line in fin ]
        H = getHomographyMatrix()
        smoothedData = detectedPlayers(players_list, H)
        
    with open(outputFilePath, 'w') as fout:
        for frameData in smoothedData:
            fout.write(str(frameData))
            fout.write("\n")

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
'''
def getDistancesWalkedFramewise(playerPos):
    playerDist = np.zeros((playerPos[0].shape[0], playerPos[0].shape[1])), np.zeros((playerPos[1].shape[0], playerPos[1].shape[1]))
    for t, team in enumerate(playerPos):
        for p, player in enumerate(team):
            for i in xrange(1,player.shape[0]):
                playerDist[t][p,i] = distanceBetweenCoordsTopDown(player[i-1], player[i]) + playerDist[t][p,i-1]
                
    return playerDist

def getHeatmaps(playerPos):
    heatmaps = []
    for t, team in enumerate(playerPos):
        teammaps = []
        for p, player in enumerate(team):
            print "Generate Heatmaps for team "+str(t)+" player "+str(p)
            cv2.imwrite("../images/team"+str(t)+"_player"+str(p)+".png", heatmap.getFieldHeatmap(player))
            #teammaps.append(heatmap.getFieldHeatmap(player))
        
        #heatmaps.append(teammaps)
        #break
    
    return heatmaps

'''
    Parameters:
        players: The framewise player data, with (x,y) coordinates
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
    if teamLeft_mostLeft > teamRight_mostLeft and teamRight_mostLeft < WIDTH_TD_IMG/2:
        # Offside on the left
        print "offside!!!"
        # Get coords of the line
        #Hinv = np.linalg.inv(H)
        topPt = getInverseTransformationCoords(Hinv,BORDER,teamLeft_mostLeft)
        bottomPt = getInverseTransformationCoords(Hinv,img.shape[0]-BORDER,teamLeft_mostLeft)
        
        # Blend the new line, such that it is transparent
        imgLeftLine = copy.copy(img) 
        cv2.line(imgLeftLine, (topPt[1],topPt[0]), (bottomPt[1],bottomPt[0]),(0,0,255),10)
        
        
    # If there is a player from the left team more right than any player from team right
    # and this player is on the right players half (right = large number)  
    if teamLeft_mostRight > teamRight_mostRight and teamLeft_mostRight > WIDTH_TD_IMG/2:
        # Offside on the right
        print "offside!!!"
        # Get coords of the line
        #Hinv = np.linalg.inv(H)
        topPt = getInverseTransformationCoords(Hinv,BORDER,teamRight_mostRight)
        bottomPt = getInverseTransformationCoords(Hinv,img.shape[0]-BORDER,teamRight_mostRight)

        # Blend the new line, such that it is transparent
        imgRightLine = copy.copy(img) 
        cv2.line(imgRightLine, (topPt[1],topPt[0]), (bottomPt[1],bottomPt[0]),(0,0,255),10)
        
    # Mostlikely therefore the first one 
    if imgRightLine is None and imgLeftLine is None:
        return img
    
    if imgRightLine is not None and imgLeftLine is not None:
        img = cv2.addWeighted(img,0.33333,imgRightLine,0.66667,0)
        return cv2.addWeighted(img,0.6,imgLeftLine,0.4,0)
    
    if imgRightLine is not None:
        return cv2.addWeighted(img,0.6,imgRightLine,0.4,0)
    
    #if imgLeftLine is not None:
    return cv2.addWeighted(img,0.6,imgLeftLine,0.4,0)
    
    #return img

if __name__ == '__main__':
    # evalMapping()
    playerDataToSmoothedTopDown("players_1533.txt", PATH_SMOOTHED_TOP_DOWN_DATA)
    #framewisePos, playersPos = getPlayerWiseTopDown()
    #getDistancesWalkedFramewise(playerPos)
    #heatmaps = getHeatmaps(playerPos)
    '''
    for t, team in enumerate(heatmaps):
        for p, player in enumerate(team):
            cv2.imwrite("../images/team"+str(t)+"_player"+str(p)+".png", player)
    '''

    '''
    with open(PATH_SMOOTHED_TOP_DOWN_DATA) as fin:
        createTopDownVideo([
            eval(line) for line in fin
        ])
    '''
