'''
Created on 03.11.2015
'''
import numpy as np
import cv2

def generateVideoComposition(team1Name, team1Color, team2Name, team2Color, playerDistances, mainVideoURI, topDownVideoURI):
    pass

def getInitVideo(team1Name, team1Color, team2Name, team2Color):
    img = np.zeros((1080,1920,3), np.uint8)
    # Header
    cv2.rectangle(img,(0,0),(1920,124), (17,17,17), cv2.cv.CV_FILLED)
    # Left bottom
    cv2.rectangle(img, (0,415),(960,1080), (17,17,17), cv2.cv.CV_FILLED)
    # Left bottom Header
    cv2.rectangle(img, (0,415),(960,505), (37,37,37), cv2.cv.CV_FILLED)
    # Left bottom Header Text
    cv2.putText(img, "Distance walked by each player", (210,470), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255)) 
    
    
    cv2.putText(img, team1Name, (720,70), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255)) 
    cv2.putText(img, team2Name, (1060,70), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255)) 
    cv2.putText(img, "vs", (940,75), cv2.FONT_HERSHEY_DUPLEX, 2.7, (255,255,255), 3)
    
    
    # Color Team 1
    cv2.rectangle(img, (680,50),(700,70), team1Color, cv2.cv.CV_FILLED)
    cv2.rectangle(img, (680,50),(700,70), (0,0,0),2)
    # Color Team 2
    cv2.rectangle(img, (1335,50),(1355,70), team2Color, cv2.cv.CV_FILLED)
    cv2.rectangle(img, (1335,50),(1355,70), (0,0,0),2)
    
    # Team header distance
    cv2.putText(img, team1Name, (170,620), cv2.FONT_HERSHEY_DUPLEX, 1, team1Color) 
    cv2.putText(img, team2Name, (550,620), cv2.FONT_HERSHEY_DUPLEX, 1, team2Color) 
    
    for i in xrange(11):
        name = "Player" + str(i+1) + ":"
        cv2.putText(img, name, (150,660+i*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
        cv2.putText(img, name, (560,660+i*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
    
    return img

def test():
    team1Name = "Warriors FC"
    team2Name = "Balestier Khalsa"
    team1Color = (255,0,0)
    team2Color = (0,0,255)
    img = getInitVideo(team1Name, team1Color, team2Name, team2Color)
    
    capMain = cv2.VideoCapture('../videos/stitched.mpeg')
    capTopDown = cv2.VideoCapture('../videos/topDown.mpeg')
    
    ## for over all frames
    # doublicate init image before writing
    ret, frame = capMain.read()
    resized_image = cv2.resize(frame, (1920, 241))
    img[149:149+241,:,:] = resized_image

    ret, frame = capTopDown.read()
    resized_image = cv2.resize(frame, (960, 665))
    img[415:,960:,:] = resized_image
    
    # Write walked distance
    for i in xrange(11):
        distance = str(i*100) + " m"
        cv2.putText(img, distance, (280,660+i*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
        cv2.putText(img, distance, (690,660+i*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
    ## endfor
    
    #cv2.imshow("img", resized_image)
    cv2.imwrite("img.jpg", img)
    cv2.waitKey(0)
    pass


    
if __name__ == '__main__':
    test()
