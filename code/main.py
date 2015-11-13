from stitch import stitch
from gen_background import genBg
import player_displayer
import player_detector
from videoComposition import createFinalVideo
from util import deleteFirstNLines


'''
    The main function call all the functions necessary to get from 
    the 3 single videos to the final composition with the detected
    players. If one is only interested in running a part, just 
    comment the other parts out.
'''
if __name__ == '__main__':
    # Stitch the video
    stitch()
    # Generate the Background
    genBg()
    # Detect the players
    player_detector.generatePlayersList()
    player_detector.generateVideoWithRect("players.txt","../videos/stitched_rect.mpeg")
    # Remove the first 4 lines of the input file
    deleteFirstNLines("players.txt", "players_wo4.txt", 4)
    # Generate the top down data
    player_displayer.playerDataToSmoothedTopDown("players_wo4.txt")
    # Generate top down video
    player_displayer.createTopDownVideo()
    # Generate heatmaps
    _, playersPos = player_displayer.getPlayerWiseTopDown()
    player_displayer.generateHeatmaps(playersPos)
    # Generate final Video
    createFinalVideo()
