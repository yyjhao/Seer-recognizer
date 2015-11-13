# Readme for the final project for module CS4243

## Create the final video
To generate the final video from only the single videos, one only needs to execute the file "main.py" it calls all required functions sequential. During the generation the will be a huge amount of png files created, these are necessary for an optimal detection. Because, the stitched.mpeg is compressed and therewith, important image information is lost. The png's provide the uncompressed information and further they have the huge advantage over a uncompressed video that they are single files which can also be generated in parallel. After the player detection, these png's can be deleted.
Besides, the initial input files, it is necessary that the following empty folders exists: "images/heatmaps/" and "images/stitched_frames/".

## List of files required and generated during execution
Here you can see which files are required to execute the main.py, which files will be generated during the execution. These file generations allows the user to run only parts of it at once and other parts later. This is usefull as some tasks take very long to run. If one would like to always run everything in one go, then it should be piped instead of written to the disk. All path are from the 'codes' folder.

### Initial input
* "../images/FootballField_small_border.png"
* "../videos/football_mid.mp4"
* "../videos/football_left.mp4"
* "../videos/football_right.mp4"

### Intermediate output

* _Compressed stitched video_
    * "../videos/stitched.mpeg"
* _Uncompressed stitched png files of each frame, for detection and background image extraction_
    * "../images/stitched_frames/{}.png".format(i)
* _Uncompressed background images required for the detection_
    * "../images/stitched_background.png"	
* _File with the position of each player in each frame_
    * "players.txt"
* _Players file without the first 4 lines_
    * "players_wo4.txt"
* _Smoothed and tracked player position_
    * "players_smoothedTopDown.txt"
* _Compressed stitched video with a detection rectangle around the players_
    * "../videos/stitched_rect.mpeg"
* _Compressed top down video of the players_
    * "../videos/topDown.mpeg"
* _Heatmaps for each player_
    * "../images/heatmaps/team" + str(teamId) + "_player" + str(playerId) + ".png"

### Final output
* "../videos/final.mpeg"