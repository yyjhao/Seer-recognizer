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

    mask = np.zeros(imgShape)
    pts[0,:] = list(reversed(pts[0,:]))
    pts[1,:] = list(reversed(pts[1,:]))
    pts[2,:] = list(reversed(pts[2,:]))
    pts[3,:] = list(reversed(pts[3,:]))
    fill = 1
    if len(imgShape) == 3:
        fill = [1 for i in range(imgShape[2])]
    cv2.fillPoly(mask, [pts], fill)
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

def rect_centroid(rect):
    """ Given a rectangle defined by rect = (x, y, w, h),
    return the nearest-int location of its centroid, as a tuple (x, y).
    """
    x, y, w, h = rect
    return (x + (w / 2), y + (h / 2))

def euclidean_distance(p1, p2):
    """ Computes the Euclidean distance between two points
    p1 = (x1, y1) and p2 = (x2, y2).
    Returns a float >= 0.
    """
    result = np.linalg.norm(np.array(p1) - np.array(p2))
    assert result >= 0
    return result

if __name__ == '__main__':
    testMask()