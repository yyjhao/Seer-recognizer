'''
Provides methods to transform any point on the original field to a point on the top down field
'''
import cv2
import numpy as np
from player_detector import getPlayers

def getHomographyMatrix():
    img = cv2.imread('../images/FootballField_small.png')#, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    # Define the 4 corner of the football field (target points), thereby the width will be kept and only the height adjusted
    # so we don't accidently lose to many information
    # [y,x]
    #FIELD_HEIGHT = 70.0
    #FIELD_WIDTH = 105.0
    #ratio = FIELD_HEIGHT/FIELD_WIDTH
    
    #newImg = np.zeros([ratio*img.shape[1], img.shape[1],3])
    
    target_pts = np.zeros([5,2])
    target_pts[0,:] = [0,0]                     # Top left (image is orientated to the top left)
    target_pts[1,:] = [0, img.shape[1]-1]        # Top right
    target_pts[2,:] = [ img.shape[0]-1, img.shape[1]-1]     # Bottom right
    target_pts[3,:] = [img.shape[0]-1, 0]     # Bottom left
    target_pts[4,:] = [img.shape[0]/2,img.shape[1]/2]     # Center points
    
    ## Points on the image
    pts = np.zeros([5,2])
    pts[0,:] = [3042,246] # x, y
    pts[1,:] = [5360, 227]
    pts[2,:] = [8680, 999] # Bottom right (outside of the image)
    pts[3,:] = [52, 1042]
    pts[4,:] = [4267, 400] # Center
    
    # Calculate the homography matrix, which will be used to project any point from the video to the top down view.
    return homography(target_pts, pts) 
    
# Transforms 2d points to 2d points on another plane
def getTransformationCoords(H, point):
    # Transform
    tmp = np.dot(H,np.array([point[0],point[1],1]))
    # Normalize
    tmp = tmp/tmp[2]
    return [int(tmp[0]), int(tmp[1])]
    
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


def test():
    img = cv2.imread('../images/FootballField_small.png')#, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    H = getHomographyMatrix()
    
    cap = cv2.VideoCapture('../videos/stitched_fixed.mpeg')
    
    i = 0
    while i < 20:
        i = i + 1
        ret, frame = cap.read()
    
    cv2.imwrite('frame20.jpg', frame)
    players = getPlayers(frame)    
    img = addPlayers(img, H, players)
    
    cap.release()
    #a = np.array([318,344,1])
    #a = np.array([225,97,1])
    cv2.imwrite('frame20_topDown.jpg', img)
    cv2.imshow('new img', img)
    cv2.waitKey(0)

# Add all players in the given players list to the field
# img: image to add the 
#
def addPlayers(img, H, players):
    a = np.zeros([2])
    i = 0
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
        except IndexError:
            print 'Player '+str(i)+' out side the field!'
            
    return img

if __name__ == '__main__':
    test()
