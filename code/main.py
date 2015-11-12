import gen_background # importing it runs the code :)
import player_displayer
import player_detector
from videoComposition import createFinalVideo
from util import deleteFirstNLines
from stitch import stitch

'''
    The main function call all the functions necessary to get from 
    the 3 single videos to the final composition with the detected
    players. If one is only interested in running a part, just 
    comment the other parts out.
'''
if __name__ == '__main__':
    # Stitch the video
    stitch()
    # Detect the players
    player_detector.generatePlayersList()
    player_detector.generateVideoWithRect("../videos/stitched.mpeg", "players_cat.txt","../videos/stitched_rect.mpeg")
    # Remove the first 4 lines of the input file
    deleteFirstNLines("players_cat.txt", "players_cat_wo4.txt", 4)
    # Generate the top down data
    player_displayer.playerDataToSmoothedTopDown("players_cat_wo4.txt")
    # Generate top down video
    player_displayer.createTopDownVideo()
    # Generate heatmaps
    _, playersPos = player_displayer.getPlayerWiseTopDown()
    player_displayer.generateHeatmaps(playersPos)
    # Generate final Video
    createFinalVideo()
