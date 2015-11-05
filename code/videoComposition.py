'''
Created on 03.11.2015
'''
import numpy as np
import cv2
import copy
import player_displayer

def generateVideoComposition(team1Name, team1Color, team2Name, team2Color, playerDistances, mainVideoURI, topDownVideoURI):
    pass

def getInitVideo(team1Name, team1Color, team2Name, team2Color):
    img = np.zeros((1080,1920,3), np.uint8)
    # Header
    cv2.rectangle(img,(0,0),(1920,124), (17,17,17), cv2.cv.CV_FILLED)
   
    
    
    cv2.putText(img, team1Name, (720,70), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255)) 
    cv2.putText(img, team2Name, (1060,70), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255)) 
    cv2.putText(img, "vs", (940,75), cv2.FONT_HERSHEY_DUPLEX, 2.7, (255,255,255), 3)
    
    return img

def createMainVideo(team1Name, team1Color, team2Name, team2Color):
    img = getInitVideo(team1Name, team1Color, team2Name, team2Color)
    # Color Team 1
    cv2.rectangle(img, (680,50),(700,70), team1Color, cv2.cv.CV_FILLED)
    cv2.rectangle(img, (680,50),(700,70), (0,0,0),2)
    # Color Team 2
    cv2.rectangle(img, (1335,50),(1355,70), team2Color, cv2.cv.CV_FILLED)
    cv2.rectangle(img, (1335,50),(1355,70), (0,0,0),2)
    
     # Left bottom
    cv2.rectangle(img, (0,415),(960,1080), (17,17,17), cv2.cv.CV_FILLED)
    # Left bottom Header
    cv2.rectangle(img, (0,415),(960,505), (37,37,37), cv2.cv.CV_FILLED)
    # Left bottom Header Text
    cv2.putText(img, "Distance walked by each player", (210,470), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255)) 
    
    # Team header distance
    cv2.putText(img, team1Name, (170,620), cv2.FONT_HERSHEY_DUPLEX, 1, team1Color) 
    cv2.putText(img, team2Name, (550,620), cv2.FONT_HERSHEY_DUPLEX, 1, team2Color) 
    
    capMain = cv2.VideoCapture('../videos/stitched.mpeg')
    capTopDown = cv2.VideoCapture('../videos/topDown.mpeg')
    
    fps = capMain.get(cv2.cv.CV_CAP_PROP_FPS)
    movie_shape = (img.shape[1], img.shape[0])
    
    fourcc = cv2.cv.CV_FOURCC(*"MPEG")
    output = cv2.VideoWriter('../videos/final.mpeg', fourcc, fps, movie_shape)

    # Get player data
    framewisePos, playersPos = player_displayer.getPlayerWiseTopDown()
    playersDist = player_displayer.getDistancesWalkedFramewise(playersPos)
    
    # Write player names (if the would change during the game, this needs to be done for each frame!)
    ## First team 1
    for i in xrange(playersPos[0].shape[0]):
        name = "Player" + str(i+1) + ":"
        cv2.putText(img, name, (150,660+i*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
    
    ## Second team 2
    for i in xrange(playersPos[1].shape[0]):
        name = "Player" + str(i+1) + ":"
        cv2.putText(img, name, (560,660+i*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
    
    # Get the inverted homography matrix for the offset detection
    H = player_displayer.getHomographyMatrix()
    Hinv = np.linalg.inv(H)
    
    ## Skip first 3 frames of the main video:
    #for i in xrange(3):
    #    _, frame = capMain.read()
            
    ## for over all frames
    # doublicate init image before writing
    for i in xrange(int(capMain.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))):
        if i>=57:
            break
        
        imgCp = copy.copy(img)
        _, frame = capMain.read()
        
        # Check for offside, if so, draw the line
        frame = player_displayer.drawOffsetLines(framewisePos[i], frame, Hinv)
        resized_image = cv2.resize(frame, (1920, 241))
        
        imgCp[149:149+241,:,:] = resized_image
    
        _, frame = capTopDown.read()
        resized_image = cv2.resize(frame, (960, 665))
        imgCp[415:,960:,:] = resized_image
        
        ## First team 1
        for p in xrange(playersPos[0].shape[0]):
            distance = '{:5.0f}'.format(playersDist[0][p,i]) + " m"
            cv2.putText(imgCp, distance, (280,660+p*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
        
        ## Second team 2
        for p in xrange(playersPos[1].shape[0]):
            distance = '{:5.0f}'.format(playersDist[1][p,i]) + " m"
            cv2.putText(imgCp, distance, (690,660+p*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
            
        
        output.write(imgCp)
        
    #cv2.imshow("img", resized_image)
    #cv2.imwrite("img.jpg", imgCp)
    
    capMain.release()
    capTopDown.release()
    output.release()
        

def generateLastFrame():
    pass

def test():
    team1Name = "Warriors FC"
    team2Name = "Balestier Khalsa"
    team1Color = (255,0,0)
    team2Color = (0,0,255)
    createMainVideo(team1Name, team1Color, team2Name, team2Color)
    
    
if __name__ == '__main__':
    test()
