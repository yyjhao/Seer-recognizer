'''

    Methods to generate a heatmap on a football field.

'''

import random
import numpy as np
import cv2
import copy

'''
    Function which draws a blur circle, the closer the pixels
    are to the center the higher their value is increased

    @param pt: points [row, col]
    @param img: reference to the image draw the circle on
    @param r: radius (int value)
'''
def blurCircle(pt, img, r):
    for col in xrange(-r, r + 1):
        height = int(np.sqrt(r * r - col * col))

        for row in xrange(-height, height + 1):
            try:
                img[row + pt[0], col + pt[1]] += (r - int(np.sqrt(row * row + col * col)));
            except IndexError:
                pass

'''
    Generates a headmap on a given image

    @param points: list with all the positions of the player in each frame as a 2d tuple/array
    @param img: The image, the heatmap should be drawn on
'''
def generateHeatmap(points, img):
    radius = 50

    for i, pt in enumerate(points):
        if i % 23 == 0:
            blurCircle(pt, img, radius)

    # Scale img to [0,255]
    img = img * (255.0 / img.max())
    img = img.astype(np.uint8)
    # Apply the colormap on the img, 255 -> red, 0->blue
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)


'''
    Draw a heat map on the top down view football field.

    @param pts: list with all the positions of the player in each frame as a 2d tuple/array

    @return The field with a heatmap drawn on
'''
def getFieldHeatmap(pts):
    field = cv2.imread('../images/FootballField_small_border.png')
    # Get random coordinates
    img = np.zeros_like(field, np.int64)
    img = generateHeatmap(pts, img)

    imgCopy = copy.copy(img)
    img[field >= 200] = 255
    img = cv2.addWeighted(img, 0.5, imgCopy, 0.5, 0)

    return img
    #cv2.imwrite("heatmap.png", img)
    #cv2.waitKey(0)

'''
    Main function to test the heatmap generation by inserting random points
'''
if __name__ == '__main__':
    width = 1400
    height = 1000
    # Get random coordinates
    pts = [[int(random.random() * height) if i == 0 else int(random.random() * width) for i in range(2)] for j in range(3000)]
    img = getFieldHeatmap(pts)
    cv2.imwrite("heatmap.png", img)
