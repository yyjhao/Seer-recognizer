'''
Created on 03.11.2015
'''
import numpy as np
import cv2
import copy
import player_displayer
from player_detector import Color

def generateVideoComposition(team1Name, team1Color, team2Name, team2Color, playerDistances, mainVideoURI, topDownVideoURI):
    pass

def getInitVideo(team1Name, team1Color, team2Name, team2Color):
    img = np.zeros((1080,1920,3), np.uint8)
    # Header
    cv2.rectangle(img,(0,0),(1920,124), (17,17,17), cv2.cv.CV_FILLED)
   
    cv2.putText(img, team1Name, (720,70), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255)) 
    cv2.putText(img, team2Name, (1060,70), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255)) 
    cv2.putText(img, "vs", (940,75), cv2.FONT_HERSHEY_DUPLEX, 2.7, (255,255,255), 3)
    
    # Color Team 1
    cv2.rectangle(img, (680,50),(700,70), team1Color, cv2.cv.CV_FILLED)
    cv2.rectangle(img, (680,50),(700,70), (0,0,0),2)
    # Color Team 2
    cv2.rectangle(img, (1335,50),(1355,70), team2Color, cv2.cv.CV_FILLED)
    cv2.rectangle(img, (1335,50),(1355,70), (0,0,0),2)
    
    return img

def createMainVideo(team1Name, team1Color, team2Name, team2Color):
    img = getInitVideo(team1Name, team1Color, team2Name, team2Color)
    
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
    
    ## Skip first 4 frames of the main video:
    for i in xrange(4):
        _, frame = capMain.read()
            
    ## for over all frames
    # doublicate init image before writing
    imgCp = None
    for i in xrange(int(capMain.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))-4):
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
            
        if (i==0):
            # Add the front page and the transition effect
            first = getIntroFrame(team1Name, team1Color, team2Name, team2Color)
            for f in xrange(23):
                output.write(first)
            frames = 23
            width = 960.0
            pxPerFrame = width/frames
            # Move the halfes of the front image to the sides
            for f in xrange(frames):
                base = copy.copy(imgCp)
                base[:,0:int((frames-f)*pxPerFrame),:] = first[:,960-int((frames-f)*pxPerFrame):960,:]
                base[:,960+int((f)*pxPerFrame):,:] = first[:,960:1920-int((f)*pxPerFrame),:]
                output.write(base)
                
        output.write(imgCp)
        
    #cv2.imshow("img", resized_image)
    #cv2.imwrite("img.jpg", imgCp)
    
    # Wait 5 more frames before blending
    for i in xrange(5):
        output.write(imgCp)
        
    black = np.zeros_like(imgCp, np.uint8)
    frames = 12
    
    # Fade to black
    for i in xrange(frames):
        output.write(cv2.addWeighted(imgCp,(frames-i)/46.0,black,i/46.0,0))
    
    # Fade to end frame
    finalImg = getLastFrame(team1Name, team1Color, team2Name, team2Color, playersPos)
    for i in xrange(frames):
        output.write(cv2.addWeighted(black,(frames-i)/46.0,finalImg,i/46.0,0))
    
    # Show the final frame for 1 second
    for i in xrange(46):
        output.write(finalImg)
    
    capMain.release()
    capTopDown.release()
    output.release()

def getIntroFrame(team1Name, team1Color, team2Name, team2Color):
    img = np.zeros((1080,1920,3), np.uint8)
    
    cv2.rectangle(img, (0,0),(960,1080), (13,13,13), cv2.cv.CV_FILLED)
    
    # (x,y)
    cv2.putText(img, team1Name, (300,280), cv2.FONT_HERSHEY_DUPLEX, 3, (255,255,255),3) 
    cv2.putText(img, team2Name, (980,560), cv2.FONT_HERSHEY_DUPLEX, 3, (255,255,255),3) 
    cv2.putText(img, "vs", (875,420), cv2.FONT_HERSHEY_DUPLEX, 5, (255,255,255), 7)
    
    # Color Team 1
    cv2.rectangle(img, (170,200),(270,300), team1Color, cv2.cv.CV_FILLED)
    cv2.rectangle(img, (170,200),(270,300), (0,0,0),2)
    # Color Team 2
    cv2.rectangle(img, (1780,480),(1880,580), team2Color, cv2.cv.CV_FILLED)
    cv2.rectangle(img, (1780,480),(1880,580), (0,0,0),2)
    
    subtitle1 = "Computer Vision Based Football Game Analysis"
    subtitle2= "by Charles L., Dennis S., Larry X., and Yujian Y."
    cv2.putText(img, subtitle1, (200,850), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255),3) 
    cv2.putText(img, subtitle2, (300,920), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255,255,255),3)
    return img        

def getLastFrame(team1Name, team1Color, team2Name, team2Color, playersPos):
    img = getInitVideo(team1Name, team1Color, team2Name, team2Color)
    
    cv2.rectangle(img, (0,124),(960,1080), (13,13,13), cv2.cv.CV_FILLED)
    
    # Write player names (if the would change during the game, this needs to be done for each frame!)
    ## First team 1
    
    for t in xrange(2):
        for i in xrange(playersPos[t].shape[0]):
            row = i / 3 # 3 Elements per row
            col = i % 3
            name = "Player" + str(i+1) + ""
            
            heatmapPath = '../images/heatmaps/team'+str(t)+'_player' + str(i) + '.png'
            heatmap = cv2.imread(heatmapPath)
            resized_image = cv2.resize(heatmap, (230, 164))
            
            cv2.putText(img, name, (t*960+90+col*(230+45)+60,124+50+row*(65+164)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
            img[124+65+row*(65+164):124+65+164+row*(65+164),t*960+90+col*(230+45):t*960+90+230+col*(230+45)] = resized_image
            #cv2.putText(img, name, (150,660+i*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))

    return img

def createFinalVideo():
    team1Name = "Warriors FC"
    team2Name = "Balestier Khalsa"
    team1Color = Color.RED
    team2Color = Color.BLUE
    createMainVideo(team1Name, team1Color, team2Name, team2Color)
    #img = getIntroFrame(team1Name, team1Color, team2Name, team2Color)
    #img = getLastFrame(team1Name, team1Color, team2Name, team2Color)
    #cv2.imwrite("first.jpg", img)
    #cv2.waitKey(0)
    
if __name__ == '__main__':
    createFinalVideo()
