import player_displayer
import player_detector
from videoComposition import createFinalVideo

if __name__ == '__main__':
    # Stitch the video
    # run stitch.py
    # Detect the players
    # player_detector.main3()
    # Generate the top down data
    player_displayer.playerDataToSmoothedTopDown("players_test7.txt")
    # Generate top down video
    player_displayer.createTopDownVideo()
    # Generate heatmaps
    _, playersPos = player_displayer.getPlayerWiseTopDown()
    player_displayer.generateHeatmaps(playersPos)
    # Generate final Video
    createFinalVideo()
    