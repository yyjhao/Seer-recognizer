'''
Provides methods to transform any point on the original field to a point on the top down field
'''
import cv2
import numpy as np

def getHomographyMatrix():
    img = cv2.imread('../images/FootballField.jpg')#, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    # Define the 4 corner of the football field (target points), thereby the width will be kept and only the height adjusted
    # so we don't accidently lose to many information
    # [y,x]
    #FIELD_HEIGHT = 70.0
    #FIELD_WIDTH = 105.0
    #ratio = FIELD_HEIGHT/FIELD_WIDTH
    
    #newImg = np.zeros([ratio*img.shape[1], img.shape[1],3])
    
    target_pts = np.zeros([7,2])
    target_pts[0,:] = [0,0]                     # Top left (image is orientated to the top left)
    target_pts[1,:] = [0,img.shape[1]-1]        # Top right
    target_pts[2,:] = [img.shape[0]-1,img.shape[1]-1]     # Bottom right
    target_pts[3,:] = [img.shape[0]-1,0]     # Bottom left
    target_pts[4,:] = [img.shape[0]/2,img.shape[1]/2]     # Center points
    target_pts[5,:] = [img.shape[0]/2,619]     # 11m right side
    target_pts[6,:] = [img.shape[0]/2,82]     # 11m left side
    
    ## Points on the image
    pts = np.zeros([7,2])
    pts[0,:] = [246, 3042] # y, x
    pts[1,:] = [227, 5360]
    pts[2,:] = [999,8680] # Bottom right (outside of the image)
    pts[3,:] = [1042, 52]
    pts[4,:] = [400, 4267] # Center
    pts[5,:] = [386, 5676] # 11m right side
    pts[6,:] = [416, 2788] # 11m left side
    
    # Calculate the homography matrix, which will be used to project any point from the video to the top down view.
    return homography(target_pts, pts) 

    '''
    print H
    
    #a = np.array([318,344,1])
    #a = np.array([225,97,1])
    a = np.dot(H,np.array([225,97,1]))
    a = a/a[2]
    print a,getTransformationCoords(H, [225,97])
    a = np.dot(H,np.array([206,632,1]))
    a = a/a[2]
    print a,getTransformationCoords(H, [206,632])
    a = np.dot(H,np.array([img.shape[0]-1,img.shape[1]-1,1]))
    a = a/a[2]
    print a,getTransformationCoords(H, [img.shape[0]-1,img.shape[1]-1])
    
    a = np.dot(H,np.array([img.shape[0]-1,0,1]))
    a = a/a[2]
    print a,getTransformationCoords(H, [img.shape[0]-1,0])
    print getTransformationCoords(H, [318,344])
    #a = np.array([206,632,1])
    #b = np.dot(H,a)
    
    
    #newImg[int(b[0]),int(b[1])] = 255
    #cv2.imshow('new img', newImg)
    #cv2.waitKey(0)
    '''
    
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
    img = cv2.imread('../images/FootballField.jpg')#, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    H = getHomographyMatrix()
    #a = np.array([318,344,1])
    #a = np.array([225,97,1])
    b = np.zeros([2,2])
    b[0,:] = getTransformationCoords(H, [416,2788])
    b[1,:] = getTransformationCoords(H, [386, 5676])
   
    img[int(b[0,0]),int(b[0,1])] = [0,0,255]
    img[int(b[1,0]),int(b[1,1])] = [0,0,255]
    cv2.imshow('new img', img)
    cv2.waitKey(0)
    pass

if __name__ == '__main__':
    test()