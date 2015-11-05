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
    fill = 1
    if len(imgShape) == 3:
        fill = [1 for i in range(imgShape[2])]
    cv2.fillConvexPoly(mask, pts, fill)
    return mask

def testMask():
    ## Corners and center
    pts = np.zeros([7,2], dtype=np.int)
    pts[0,:] = [2592, 199] # Top left
    pts[1,:] = [4892, 182] # Top right
    pts[2,:] = [5400, 288] # Goalie top left
    pts[3,:] = [5948, 288] # Goalie top right
    pts[4,:] = [5948, 408] # Goalie bottom right
    pts[5,:] = [8500, 990] # Bottom right
    pts[6,:] = [-100, 990] # Bottom left

    img = cv2.imread('../images/stitched_background.png')

    mask = quadrangleMask(pts, img.shape)

    cv2.imwrite('../images/mask.png', img*mask)

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

def dist_point_to_rect(point, rect):
    """ Computes the shortest distance from a point to any edge of the
    rectangle. If the point is within the rectangle, distance is 0.
    Otherwise, the distance is manhattan.
    Returns an int >= 0.
    """
    px, py = point
    rx, ry, rw, rh = rect

    dist_x1 = max(0, rx - px)
    dist_x2 = max(0, px - (rx + rw))
    dist_y1 = max(0, ry - py)
    dist_y2 = max(0, py - (ry + rh))

    dist_x = max(dist_x1, dist_x2)
    dist_y = max(dist_y1, dist_y2)
    return dist_x + dist_y

if __name__ == '__main__':
    testMask()