'''
Provides methods to transform any point on the original field to a point on the top down field
'''
import cv2
import numpy as np
import copy
from player_detector import getPlayers

BORDER = 50

def getHomographyMatrix():
    img = cv2.imread('../images/FootballField_small_border.png')#, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    # Define the 4 corner of the football field (target points), thereby the width will be kept and only the height adjusted
    # so we don't accidently lose to many information
    # [y,x]
    #FIELD_HEIGHT = 70.0
    #FIELD_WIDTH = 105.0
    #ratio = FIELD_HEIGHT/FIELD_WIDTH
    
    #newImg = np.zeros([ratio*img.shape[1], img.shape[1],3])
    target_pts = np.zeros([21,2])
    # y,x
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
    
    
    
    ## Points on the image
    pts = np.zeros([21,2])
    ## Special points on the image
    pts[0,:] = [2785,355] #a
    pts[1,:] = [2279, 504]
    pts[2,:] = [3305, 297]
    pts[3,:] = [2380, 656]
    pts[4,:] = [3151, 407]
    pts[5,:] = [3955, 399]
    pts[6,:] = [4580, 399]
    pts[7,:] = [5386, 397]
    pts[8,:] = [5152, 280]
    pts[9,:] = [6174, 624]
    pts[10,:] = [5645, 329]
    pts[11,:] = [6182, 468]
    pts[12,:] = [5802, 327]
    pts[13,:] = [6405, 467]
    pts[14,:] = [2613, 355]
    pts[15,:] = [2037, 508] #p
    
    ## Corners and center
    pts[16,:] = [3042,247] # x, y
    pts[17,:] = [5360, 227]
    pts[18,:] = [8680, 999] # Bottom right (outside of the image)
    pts[19,:] = [50, 1042]
    pts[20,:] = [4267, 400] # Center
    
    
    
    # Calculate the homography matrix, which will be used to project any point from the video to the top down view.
    return homography(target_pts, pts) 
    
# Transforms 2d points to 2d points on another plane
def getTransformationCoords(H, point):
    # Transform
    tmp = np.dot(H,np.array([point[0],point[1],1]))
    # Normalize
    tmp = tmp/tmp[2]
    return [int(tmp[0]), int(tmp[1])]
    
def getInverseTransformationCoords(Hinv, x, y):
    tmp = np.dot(Hinv,np.array([y,x,1]))
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

def evalMapping():
    img = cv2.imread('../images/FootballField_small_border.png')#, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    H = getHomographyMatrix()
    
    # Define relevant points on the original field
    pts = np.zeros([21,2])
    pts[0,:] = [2785,355]
    pts[1,:] = [2279, 504]
    pts[2,:] = [3305, 297]
    pts[3,:] = [2380, 656]
    pts[4,:] = [3151, 407]
    pts[5,:] = [3955, 399]
    pts[6,:] = [4580, 399]
    pts[7,:] = [5386, 397]
    pts[8,:] = [5152, 280]
    pts[9,:] = [6174, 624]
    pts[10,:] = [5645, 329]
    pts[11,:] = [6182, 468]
    pts[12,:] = [5802, 327]
    pts[13,:] = [6405, 467]
    pts[14,:] = [2613, 355]
    pts[15,:] = [2037, 508]
    
    # Corners and center
    pts[16,:] = [3042,246] # x, y
    pts[17,:] = [5360, 227]
    pts[18,:] = [8680, 999] # Bottom right (outside of the image)
    pts[19,:] = [52, 1042]
    pts[20,:] = [4267, 400] # Center
    
    a = np.zeros([2])
    i = 0
    for pt in pts:
        i = i + 1
        a = getTransformationCoords(H, [pt[0],pt[1]])
        try:
            for i in xrange(5):
                for j in xrange(5):
                    if i+j <= 5 :
                        img[int(a[0]+i),int(a[1]+j)] = [0,0,255]
                        img[int(a[0]+i),int(a[1]-j)] = [0,0,255]
                        img[int(a[0]-i),int(a[1]+j)] = [0,0,255]
                        img[int(a[0]-i),int(a[1]-j)] = [0,0,255]
        except IndexError:
            print 'Player '+str(i)+' out side the field!'
    
    cv2.imwrite('EvalField.jpg', img)        
    cv2.imshow('new img', img)
    cv2.waitKey(0)
    

def test():
    img = cv2.imread('../images/FootballField_small_border.png')#, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    H = getHomographyMatrix()
    
    cap = cv2.VideoCapture('../videos/stitched_fixed.mpeg')
    
    i = 0
    while i < 10:
        i = i + 1
        ret, frame = cap.read()
    
    
    players = getPlayers(frame)    
    playersOnTheField, img = addPlayers(img, H, players)
    
    cap.release()
    #a = np.array([318,344,1])
    #a = np.array([225,97,1])
    cv2.imwrite('frame10_topDown.jpg', img)
    #cv2.imshow('new img', img)
    
    # blue is on the right side
    # red is on the left side (defined by user)
    # Colors of players on the left and right side - goalies should have a slightly other color!
    left = (0,0,255)
    right = (255,0,0)
    teamLeft_mostLeft = 9999999
    teamLeft_mostRight = -1
    teamRight_mostLeft = 9999999
    teamRight_mostRight = -1
    
    for player in playersOnTheField:
        if player[0][1] == left:
            if player[1][0] < teamLeft_mostLeft:
                teamLeft_mostLeft = player[1][0]
            if player[1][0] > teamLeft_mostRight:
                teamLeft_mostRight = player[1][0]
        elif player[0][1] == right:
            if player[1][0] < teamRight_mostLeft:
                teamRight_mostLeft = player[1][0]
            if player[1][0] > teamRight_mostRight:
                teamRight_mostRight = player[1][0]
            
    if teamRight_mostLeft < teamLeft_mostLeft:
        print "offside!!!"
    if teamLeft_mostRight > teamRight_mostRight:
        print "offside!!!"
    
    # Draw offside line on both sides if the last player of each team is in his half
    if teamRight_mostRight > img.shape[1]/2:
        # Draw line
        Hinv = np.linalg.inv(H)
        topPt = getInverseTransformationCoords(Hinv,teamRight_mostRight,BORDER)
        bottomPt = getInverseTransformationCoords(Hinv,teamRight_mostRight,img.shape[0]-BORDER)

        frameWithLine = copy.copy(frame) 
        cv2.line(frameWithLine, topPt, bottomPt,(0,0,255),10)
        frame = cv2.addWeighted(frame,0.8,frameWithLine,0.2,0)

    if teamLeft_mostLeft < img.shape[1]/2:
        # Draw line
        Hinv = np.linalg.inv(H)
        topPt = getInverseTransformationCoords(Hinv,teamLeft_mostLeft,BORDER)
        bottomPt = getInverseTransformationCoords(Hinv,teamLeft_mostLeft,img.shape[0]-BORDER)
        frameWithLine = copy.copy(frame) 
        # Draw a solid line on the copy
        cv2.line(frameWithLine, topPt, bottomPt,(0,0,255),10)
        # Blend both image to make the line transparent on the frame
        frame = cv2.addWeighted(frame,0.8,frameWithLine,0.2,0)
    
    cv2.imwrite('frame10.jpg', frame)
    cv2.waitKey(0)

# Add all players in the given players list to the field
# img: image to add the 
def addPlayers(img, H, players):
    a = np.zeros([2])
    i = 0
    playersOnTheField = []
    for player in players:
        i = i + 1
        x = player[0][0]+player[0][2]/2.0
        y = player[0][1]+player[0][3]
        a = getTransformationCoords(H, [x,y])
        try:
            for i in xrange(5):
                for j in xrange(5):
                    if i+j <= 5 :
                        img[int(a[0]+i),int(a[1]+j)] = player[1]
                        img[int(a[0]+i),int(a[1]-j)] = player[1]
                        img[int(a[0]-i),int(a[1]+j)] = player[1]
                        img[int(a[0]-i),int(a[1]-j)] = player[1]
            # If we reach that point the player was somewhere on the field
            playersOnTheField.append((player, (a[1],a[0])))
        except IndexError:
            print 'Player '+str(i)+' out side the field!'
            
    return playersOnTheField, img

if __name__ == '__main__':
    test()
    #evalMapping()
