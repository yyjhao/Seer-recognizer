'''
Generates a video based on player.txt
'''
import cv2
import numpy as np
import copy
from player_tracker import PlayerTracker

BORDER = 50
PATH_TOP_DOWN_IMG = '../images/FootballField_small_border.png'
WIDTH_TD_IMG = 1300
HEIGHT_TD_IMG = 900

# Dimensions of the field in m without borders
FIELD_HEIGHT = 70.0
FIELD_WIDTH = 105.0

# Distance of one pixel
PX_TO_M = FIELD_WIDTH / (WIDTH_TD_IMG - 2*BORDER) 

def getHomographyMatrix():
    img = cv2.imread(PATH_TOP_DOWN_IMG)#, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    # Define the 4 corner of the football field (target points), thereby the width will be kept and only the height adjusted
    # so we don't accidently lose to many information
    # [y,x]
    
    
    #newImg = np.zeros([ratio*img.shape[1], img.shape[1],3])
    target_pts = np.zeros([21,2])
    # y,x (row, col)
    target_pts[0,:] = [BORDER+295,BORDER+69]
    target_pts[1,:] = [BORDER+505,BORDER+69]
    target_pts[2,:] = [BORDER+170,BORDER+190]
    target_pts[3,:] = [BORDER+630,BORDER+190]
    target_pts[4,:] = [BORDER+400,BORDER+230]
    target_pts[5,:] = [BORDER+400,BORDER+494]
    target_pts[6,:] = [BORDER+400,BORDER+706]
    target_pts[7,:] = [BORDER+400,BORDER+970]
    target_pts[8,:] = [BORDER+170,BORDER+1010]
    target_pts[9,:] = [BORDER+630,BORDER+1010]
    target_pts[10,:] = [BORDER+295,BORDER+1131]
    target_pts[11,:] = [BORDER+505,BORDER+1131]
    target_pts[12,:] = [BORDER+295,BORDER+1195]
    target_pts[13,:] = [BORDER+505,BORDER+1195]
    target_pts[14,:] = [BORDER+295,BORDER+5]
    target_pts[15,:] = [BORDER+505,BORDER+5]
    ## Corners and center
    target_pts[16,:] = [BORDER,BORDER]                     # Top left (image is orientated to the top left)
    target_pts[17,:] = [BORDER, img.shape[1]-1-BORDER]        # Top right
    target_pts[18,:] = [ img.shape[0]-1-BORDER, img.shape[1]-1-BORDER]     # Bottom right
    target_pts[19,:] = [img.shape[0]-1-BORDER, BORDER]     # Bottom left
    target_pts[20,:] = [(img.shape[0])/2,img.shape[1]/2]     # Center points
    
    
    
    ## Points on the image (row, col) == (y,x)
    pts = np.zeros([21,2])
    ## Special points on the image
    pts[0,:] = [300,2370] # A (in the white)
    pts[1,:] = [443,1929] # B (in the white)
    pts[2,:] = [245,2817] # C (in the white)
    pts[3,:] = [591,2000] # D (in the white)
    pts[4,:] = [358,2664] # E (in the white)
    pts[5,:] = [359,3454] # F (in the white)
    pts[6,:] = [355,4083] # G (in the white)
    pts[7,:] = [340,4880] # H (in the white)
    pts[8,:] = [231,4678] # I (in the white)
    pts[9,:] = [576,5705] # J (in the white)
    pts[10,:] = [280,5176] # K (in the white)
    pts[11,:] = [421,5722] # L (in the white)
    pts[12,:] = [279,5341] # M (in the white)
    pts[13,:] = [420,5948] # N (in the white)
    pts[14,:] = [301,2219] # O (in the white)
    pts[15,:] = [445,1721] # P (in the white)
        
    ## Corners and center
    pts[16,:] = [196,2593] # Top left (outer coord)
    pts[17,:] = [177,4892] # Top right (outer coord)
    pts[18,:] = [950,8206] # Bottom right (outer coord)
    pts[19,:] = [942,40] # Bottom left (outer coord)
    pts[20,:] = [350,3767] # Center
    
    
    
    # Calculate the homography matrix, which will be used to project any point from the video to the top down view.
    return homography(target_pts, pts) 
    
# Transforms 2d points to 2d points on another plane
def getTransformationCoords(H, point):
    # Transform
    tmp = np.dot(H,np.array([point[0],point[1],1]))
    # Normalize
    tmp = tmp/tmp[2]
    return [int(tmp[0]), int(tmp[1])]
    
def getInverseTransformationCoords(Hinv, row, col):
    tmp = np.dot(Hinv,np.array([row,col,1]))
    tmp = tmp/tmp[2];
    return (int(tmp[0]), int(tmp[1]))
        
# I assume projPts=u_p,v_p, pts=u_c,v_c
def homography(target_pts, source_pts):
    A = np.zeros([target_pts.shape[0]*2,9])
    
    # Fill A
    for i in xrange(target_pts.shape[0]):
        A[2*i,:] = [source_pts[i,0], source_pts[i,1], 1, 0, 0, 0, -target_pts[i,0]*source_pts[i,0], -target_pts[i,0]*source_pts[i,1],-target_pts[i,0]]
        A[2*i+1,:] = [0,0,0,source_pts[i,0],source_pts[i,1],1,-target_pts[i,1]*source_pts[i,0],-target_pts[i,1]*source_pts[i,1],-target_pts[i,1]]
    
    # Solve A and extract H
    _,_,V = np.linalg.svd(A) # U,S,V
    H = V[8].reshape((3,3))    

    # Normalize H and return the matrix
    return H / H[2][2]

def createTopDownVideo(players_list):
    img = cv2.imread(PATH_TOP_DOWN_IMG)#, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    H = getHomographyMatrix()
    
    cap = cv2.VideoCapture('../videos/stitched.mpeg')

    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    movie_shape = (img.shape[1], img.shape[0])

    fourcc = cv2.cv.CV_FOURCC(*"MPEG")
    output = cv2.VideoWriter('./topDown.mpeg', fourcc, fps, movie_shape)
    
    for i, players in enumerate(detectedPlayers(players_list, H)):
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
        smooth([project_point(point, H) for point in p.centroids])
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
    a = getTransformationCoords(H, [row,col])
    return (a[1], a[0])


def smooth_num(num_list):
    weight = 0.95
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
                    if i+j <= 5 :
                        img[int(a[0]+i),int(a[1]+j)] = player[1]
                        img[int(a[0]+i),int(a[1]-j)] = player[1]
                        img[int(a[0]-i),int(a[1]+j)] = player[1]
                        img[int(a[0]-i),int(a[1]-j)] = player[1]
            # If we reach that point the player was somewhere on the field
            playersOnTheField.append((player, (a[0],a[1])))
        except IndexError:
            print 'Player '+ str(ind) +' out side the field!'
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
    distance = np.sqrt(np.sum(np.square(pos1-pos2)))
        
    return distance * PX_TO_M


if __name__ == '__main__':
    with open("players.txt") as fin:
        createTopDownVideo([
            eval(line) for line in fin
        ])
