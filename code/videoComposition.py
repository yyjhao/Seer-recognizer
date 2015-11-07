'''

    Provides the methods to composite the final video, such as generation of the 
    intro and outro image, blending effects between them and the main video, and t
    he main video itself.

'''
import numpy as np
import cv2
import copy
import player_displayer
from player_detector import Color

'''
    Generates the base frame which is used for the main video and the outro.
    
    @param team1Name: The name of the left team (String)
    @param team1Color: The color of the left team (tupel with 3 integers)
    @param team1Name: The name of the right team (String)
    @param team1Color: The color of the right team (tupel with 3 integers)
    
    @return: the final image in 1080p resolution (1920x1080)
'''
def getInitVideo(team1Name, team1Color, team2Name, team2Color):
    # ========== Initialize the image ========== #
    img = np.zeros((1080,1920,3), np.uint8)
    
    # ========== Draw the header ========== #
    # Draw the header container
    cv2.rectangle(img,(0,0),(1920,124), (17,17,17), cv2.cv.CV_FILLED)
    # Draw the names of the teams and the "vs"
    cv2.putText(img, team1Name, (720,70), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255)) 
    cv2.putText(img, team2Name, (1060,70), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255)) 
    cv2.putText(img, "vs", (940,75), cv2.FONT_HERSHEY_DUPLEX, 2.7, (255,255,255), 3)
    
    # Draw the color box for team 1
    cv2.rectangle(img, (680,50),(700,70), team1Color, cv2.cv.CV_FILLED)
    cv2.rectangle(img, (680,50),(700,70), (0,0,0),2)
    # Draw the color box for team 2
    cv2.rectangle(img, (1335,50),(1355,70), team2Color, cv2.cv.CV_FILLED)
    cv2.rectangle(img, (1335,50),(1355,70), (0,0,0),2)
    
    return img

'''
    Generates the full video including the intro and outro.
    The main video consists of the stitched video, the topDown video as well as
    a list of how much the players ran. Further, the offside line is drawn into
    the stitched video, if the situation occurs.
    
    @param team1Name: The name of the left team (String)
    @param team1Color: The color of the left team (tupel with 3 integers)
    @param team1Name: The name of the right team (String)
    @param team1Color: The color of the right team (tupel with 3 integers)
'''
def createFullVideo(team1Name, team1Color, team2Name, team2Color):
    # Get the initial image
    img = getInitVideo(team1Name, team1Color, team2Name, team2Color)
    
    # ========== Draw distance box ========== #
    # Draw the container
    cv2.rectangle(img, (0,415),(960,1080), (17,17,17), cv2.cv.CV_FILLED)
    # Draw the header of the container
    cv2.rectangle(img, (0,415),(960,505), (37,37,37), cv2.cv.CV_FILLED)
    # Write the title
    cv2.putText(img, "Distance walked by each player", (210,470), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255)) 
    # Write the names of the playing teams
    cv2.putText(img, team1Name, (170,620), cv2.FONT_HERSHEY_DUPLEX, 1, team1Color) 
    cv2.putText(img, team2Name, (550,620), cv2.FONT_HERSHEY_DUPLEX, 1, team2Color) 
    
    # ========== Get the video streams and init the output video ========== #
    capMain = cv2.VideoCapture('../videos/stitched_rect.mpeg')
    capTopDown = cv2.VideoCapture('../videos/topDown.mpeg')
    
    fps = capMain.get(cv2.cv.CV_CAP_PROP_FPS)
    movie_shape = (img.shape[1], img.shape[0])
    
    fourcc = cv2.cv.CV_FOURCC(*"MPEG")
    output = cv2.VideoWriter('../videos/final.mpeg', fourcc, fps, movie_shape)

    # ========== Get player data ========== #
    framewisePos, playersPos = player_displayer.getPlayerWiseTopDown()
    playersDist = player_displayer.getDistancesWalkedFramewise(playersPos)
    
    # ========== Write the player names for the distance list ========== #
    # If names the would change during the game, this needs to be done for each frame (in the loops below)
    # First team 1
    for i in xrange(playersPos[0].shape[0]):
        name = "Player" + str(i+1) + ":"
        cv2.putText(img, name, (150,660+i*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
    
    # Second team 2
    for i in xrange(playersPos[1].shape[0]):
        name = "Player" + str(i+1) + ":"
        cv2.putText(img, name, (560,660+i*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
    
    # ========== Get the homography matrix ========== #
    H = player_displayer.getHomographyMatrix()
    # Invert the homography matrix
    Hinv = np.linalg.inv(H)
    
    # ========== Skip first 4 frames of the stitched video ========== #
    for i in xrange(4):
        _, frame = capMain.read()
            
    # ========== Iterate of all frames and create the main video ========== #
    imgCp = None
    for i in xrange(int(capMain.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))-4):
        print "doing", i
        if i > 6953:
            break
        # Copying the init video
        imgCp = copy.copy(img)
        
        # Read the stitched video
        _, frame = capMain.read()
        # Check for offside, if so, draw the line
        frame = player_displayer.drawOffsetLines(framewisePos[i], frame, Hinv)
        # Resize the stitched video to fit the final video
        resized_image = cv2.resize(frame, (1920, 241))
        # Place the stitched video on the video
        imgCp[149:149+241,:,:] = resized_image
    
        # Read a new frame from the top down video
        _, frame = capTopDown.read()
        # Resize the frame to fit it's position
        resized_image = cv2.resize(frame, (960, 665))
        # Place the resized frame
        imgCp[415:,960:,:] = resized_image
        
        # Write the distance of the players of team 1
        for p in xrange(playersPos[0].shape[0]):
            distance = '{:5.0f}'.format(playersDist[0][p,i]) + " m"
            cv2.putText(imgCp, distance, (280,660+p*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
        
        # Write the distance of the players of team 2
        for p in xrange(playersPos[1].shape[0]):
            distance = '{:5.0f}'.format(playersDist[1][p,i]) + " m"
            cv2.putText(imgCp, distance, (690,660+p*30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
        
        # If it is the first frame, draw the init video first, this has to be done here,
        # because the transition effect requires the first frame of the main video.
        if (i==0):
            # Get the intro frame and write him, such that it stay for 1 sec (23 frames)
            first = getIntroFrame(team1Name, team1Color, team2Name, team2Color)
            for f in xrange(23):
                output.write(first)
            frames = 23
            width = 960.0
            pxPerFrame = width/frames
            # Create the transition effect: move the halves of the intro image to the sides such as a curtain
            for f in xrange(frames):
                base = copy.copy(imgCp)
                base[:,0:int((frames-f)*pxPerFrame),:] = first[:,960-int((frames-f)*pxPerFrame):960,:]
                base[:,960+int((f)*pxPerFrame):,:] = first[:,960:1920-int((f)*pxPerFrame),:]
                output.write(base)
        
        # Write the frame of the main video    
        output.write(imgCp)
        
        
        
    # ========== Handle the end of the video ========== #
    # Wait 5 more frames before blending
    for i in xrange(5):
        output.write(imgCp)
        
    black = np.zeros_like(imgCp, np.uint8)
    frames = 12
    
    # Fade to black
    for i in xrange(frames):
        output.write(cv2.addWeighted(imgCp,(frames-i)/46.0,black,i/46.0,0))
    
    # Fade to the outro frame
    finalImg = getLastFrame(team1Name, team1Color, team2Name, team2Color, playersPos)
    for i in xrange(frames):
        output.write(cv2.addWeighted(black,(frames-i)/46.0,finalImg,i/46.0,0))
    
    # Show the outro frame for 2 seconds
    for i in xrange(46):
        output.write(finalImg)
    
    # Release all video captures
    capMain.release()
    capTopDown.release()
    output.release()

'''
    Generates the intro image, showing the playing teams as well as the developer names.
    
    @param team1Name: The name of the left team (String)
    @param team1Color: The color of the left team (tupel with 3 integers)
    @param team1Name: The name of the right team (String)
    @param team1Color: The color of the right team (tupel with 3 integers)
    
    @return: the final image in 1080p resolution (1920x1080)
'''
def getIntroFrame(team1Name, team1Color, team2Name, team2Color):
    # ========== Initialize the image ========== #
    img = np.zeros((1080,1920,3), np.uint8)
    
    # ========== Draw the container over the left half ========== #
    cv2.rectangle(img, (0,0),(960,1080), (13,13,13), cv2.cv.CV_FILLED)
    
    # ========== Draw the names of the playing teams, their colors, and a "vs" ========== #
    cv2.putText(img, team1Name, (300,280), cv2.FONT_HERSHEY_DUPLEX, 3, (255,255,255),3) 
    cv2.putText(img, team2Name, (980,560), cv2.FONT_HERSHEY_DUPLEX, 3, (255,255,255),3) 
    cv2.putText(img, "vs", (875,420), cv2.FONT_HERSHEY_DUPLEX, 5, (255,255,255), 7)
    
    # Draw the color box for team 1
    cv2.rectangle(img, (170,200),(270,300), team1Color, cv2.cv.CV_FILLED)
    cv2.rectangle(img, (170,200),(270,300), (0,0,0),2)
    # Draw the color box for team 2
    cv2.rectangle(img, (1780,480),(1880,580), team2Color, cv2.cv.CV_FILLED)
    cv2.rectangle(img, (1780,480),(1880,580), (0,0,0),2)
    
    # ========== Draw subtitles as developer names ========== #
    subtitle1 = "Computer Vision Based Football Game Analysis"
    subtitle2= "by Charles L., Dennis S., Larry X., and Yujian Y."
    cv2.putText(img, subtitle1, (200,850), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255),3) 
    cv2.putText(img, subtitle2, (300,920), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255,255,255),3)
    return img        

'''
    Generates the the frame of the full video, showing the heatmap of each player.
    
    @param team1Name: The name of the left team (String)
    @param team1Color: The color of the left team (tupel with 3 integers)
    @param team1Name: The name of the right team (String)
    @param team1Color: The color of the right team (tupel with 3 integers)
    
    @return: the final image in 1080p resolution (1920x1080)
'''
def getLastFrame(team1Name, team1Color, team2Name, team2Color, playersPos):
    # ========== Get the init video frame ========== #
    img = getInitVideo(team1Name, team1Color, team2Name, team2Color)
    # ========== Draw the container for the left team ========== #
    cv2.rectangle(img, (0,124),(960,1080), (13,13,13), cv2.cv.CV_FILLED)
    
    # ========== Write the player names and draw the heatmaps ========== #
    for t in xrange(2):
        for i in xrange(playersPos[t].shape[0]):
            row = i / 3 # 3 Elements per row
            col = i % 3
            
            # Draw the player names
            name = "Player" + str(i+1) + ""
            cv2.putText(img, name, (t*960+90+col*(230+45)+60,124+50+row*(65+164)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255))
            
            # Open the heapmap, resize it and place it on the frame
            heatmapPath = '../images/heatmaps/team'+str(t)+'_player' + str(i) + '.png'
            heatmap = cv2.imread(heatmapPath)
            resized_image = cv2.resize(heatmap, (230, 164))
            img[124+65+row*(65+164):124+65+164+row*(65+164),t*960+90+col*(230+45):t*960+90+230+col*(230+45)] = resized_image

    return img

'''
    Generates the final video, by calling "createFullVideo" with the parameters for this game.
'''
def createFinalVideo():
    team1Name = "Warriors FC"
    team2Name = "Balestier Khalsa"
    team1Color = Color.RED
    team2Color = Color.BLUE
    createFullVideo(team1Name, team1Color, team2Name, team2Color)
    #img = getIntroFrame(team1Name, team1Color, team2Name, team2Color)
    #img = getLastFrame(team1Name, team1Color, team2Name, team2Color)
    #cv2.imwrite("first.jpg", img)
    #cv2.waitKey(0)
    
if __name__ == '__main__':
    createFinalVideo()
