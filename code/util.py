import numpy as np
import cv2

'''
    Parameters:
        pts: 4x2 numpy array (e.g. np.zeros([4,2])) 
                0: top left (row, col) == (y,x)
                1: top right (row, col) == (y,x)
                2: bottom right (row, col) == (y,x)
                3: bottom left (row, col) == (y,x)
        imgShape: tuple with the shape of the img (e.g. img.shape)
'''
def quadrangleMask(pts, imgShape):
    # All the '+1' in this method are necessary to ensure that the given 4 points won't be removed.
    
    mask = np.ones(imgShape)
    
    # Set corners to zero
    ## Top left
    for i in xrange(pts[0,0]): #row
        for j in xrange(pts[0,1]): #col
            mask[i,j] = 0
            
    ## Top right
    for i in xrange(pts[1,0]): #row
        for j in xrange(pts[1,1]+1, mask.shape[1]): #col
            mask[i,j] = 0
    
    ## Bottom right
    for i in xrange(pts[2,0]+1, mask.shape[0]): #row
        for j in xrange(pts[2,1]+1, mask.shape[1]): #col
            mask[i,j] = 0

    ## Bottom right
    for i in xrange(pts[3,0]+1, mask.shape[0]): #row
        for j in xrange(pts[3,1]): #col
            mask[i,j] = 0
            
    ## above A - B    - col can iterate fix - row needs to be checked for each col
    end = pts[1,1]+1 if pts[1,1]+1 <= imgShape[1] else imgShape[1]
    for j in xrange(pts[0,1], end): # col
        # Calculate the number of rows
        max_i = int(((j-pts[0,1]) * (pts[1,0] - pts[0,0])) / ((pts[1,1]-pts[0,1])*1.0) + pts[0,0])
        if max_i > imgShape[0]: max_i = imgShape[0]
        for i in xrange(max_i):
            # This try-except is necessary because the corners of the field may lay outside of the image
            mask[i,j] = 0

    ## right of B - C    - row can iterate fix - col needs to be checked for each row
    end = pts[2,0]+1 if pts[2,0]+1 <= imgShape[0] else imgShape[0]
    for i in xrange(pts[1,0], end): # row
        # Calculate the number of rows
        max_j = int(((i-pts[1,0]) * (pts[2,1] - pts[1,1])) / ((pts[2,0]-pts[1,0])*1.0) + pts[1,1])
        #if max_j > imgShape[1]: max_j = imgShape[1]
        for j in xrange(max_j, imgShape[1]):
            # This try-except is necessary because the corners of the field may lay outside of the image
            mask[i,j] = 0
    
    ## below C - D    - col can iterate fix - row needs to be checked for each col
    end = pts[2,1]+1 if pts[2,1]+1 <= imgShape[1] else imgShape[1]
    for j in xrange(pts[3,1], end): # col
        # Calculate the number of rows
        max_i = int(((j-pts[2,1]) * (pts[3,0] - pts[2,0])) / ((pts[3,1]-pts[2,1])*1.0) + pts[2,0])
        #if max_i > imgShape[0]: max_i = imgShape[0]
        for i in xrange(max_i, imgShape[0]):
            # This try-except is necessary because the corners of the field may lay outside of the image
            mask[i,j] = 0
            
    ## left of D - A    - row can iterate fix - col needs to be checked for each row
    ### This is necessary because the corners of the field may lay outside of the image
    end = pts[3,0]+1 if pts[3,0]+1 <= imgShape[0] else imgShape[0]
    for i in xrange(pts[0,0], end): # row
        # Calculate the number of rows
        max_j = int(((i-pts[0,0]) * (pts[3,1] - pts[0,1])) / ((pts[3,0]-pts[0,0])*1.0) + pts[0,1])
        if max_j > imgShape[1]: max_j = imgShape[1]
        for j in xrange(max_j):
            mask[i,j] = 0
    
    return mask

def testMask():
    ## Corners and center
    pts = np.zeros([4,2], dtype=np.int) 
    pts[0,:] = [207, 3042] # Top left
    pts[1,:] = [187, 5360] # Top right
    pts[2,:] = [999,8680] # Bottom right (outside of the image)
    pts[3,:] = [1042, 50] # Bottom left
    
    img = cv2.imread('../images/stitched_background.jpg')
    
    mask = quadrangleMask(pts, img.shape)
    
    cv2.imwrite('mask.jpg', mask*255)
    
if __name__ == '__main__':
    testMask()