#!/usr/bin/env python

# Implementation of player tracking for final project.
# CS 4243, Computer Vision and Pattern Recognition, November 2015.
# Xiayue (Charles) Lin, A0145415L

# -- Task List --
# IDEA: instead of arbitrary search radius, use the size of the last rectangle
# (or minimum k). This will actually be necessary for detecting collision
# resolution.
# TODO: need to deal with resolution ambiguity in >2 player collisions. Need to
# track expected position through the collision so when we see a resolution, we
# know which player left.
# IDEA: use position extrapolation to deal with large area collisions.
# unnecessary to track the player to the collision if we are already
# extrapolating his position for resolution anyway.

# Fix extrapolation (should be some sort of direction of moving average,
# which we will have to start recording)
# Implement collision -> (collision, collision) resolution as described
# in comments.
# Manually check and tweak search radiuses (add imcrementing if necessary).
# After that, it should actually be all good...


import cv2
import numpy as np
import sys

from player_detector import (
    Color,
    getPlayers,
)
from util import (
    dist_point_to_rect,
    euclidean_distance,
    rect_centroid,
)

def test():
    if len(sys.argv) != 3:
        print "Usage: python %s <start_frame> <num_frames>" % sys.argv[0]
        sys.exit()
    NUM_SKIP = int(sys.argv[1])
    NUM_FRAMES = int(sys.argv[2])

    with open("players_test3.txt") as fin:
        playersList = (eval(line) for line in fin)
        # XXX: no need to skip after next pull!

    # cap = cv2.VideoCapture('../videos/stitched.mpeg')
        for i in range(NUM_SKIP):
            _ = playersList.next()
        players = playersList.next()
        pt = PlayerTracker(players)
        print "Processed frame 0."
        i = 1
        while i < NUM_FRAMES:
            rects = playersList.next()
            pt.feed_rectangles(rects)
            print "Processed frame %r." % i
            i += 1

    #del cap

    cap = cv2.VideoCapture('../videos/stitched.mpeg')
    for i in range(NUM_SKIP):
        _, _ = cap.read()
    for i in range(NUM_FRAMES):
        _, frame = cap.read()
        print "Frame %r:" % i
        # Draw the detection rectangles.
        for detection in pt.detections_by_frame[i]:
            x, y, w, h = detection.rectangle
            color = detection.color
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 1)

        for player in pt.players:
            rect = player.rectangles[i]
            if rect is not None:
                loc = player.centroids[i]
                # print loc
                cv2.circle(frame, loc, 5, player.color, -1)
                cv2.putText(frame, str(player.pid), loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
            else:
                loc = player.centroids[i]
                print "Player %r: %r (extrapolated)" % (player.pid, loc)
                cv2.circle(frame, loc, 5, (255, 255, 255), -1)
                cv2.putText(frame, str(player.pid), loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
        cv2.imwrite('../images/player_tracking/%s.png' % i, frame)



class PlayerTracker(object):
    """ Feed this class player detection data, and it will track and record
    players and their positions.

    How to use: XXX
    """
    def __init__(self, player_list):
        """ Initialize the list of players.
        player_list: [((x, y, w, h), team), ...]
        One player will be created for each entry in `player_list`, with the player's
        first frame detection data being set to the provided rectangle.
        """
        # Detection data by frame, as a list of Detection objects.
        # We store this for easier debugging and analysis.
        self.detections_by_frame = []

        detections = [Detection(x[0], x[1]) for x in player_list]
        self.detections_by_frame.append(detections)

        # The players being tracked, as a list of Player objects.
        self.players = []
        i = 0
        for i in range(len(detections)):
            detection = detections[i]
            player = Player(i, detection.color)
            player.set_next_location(detection.centroid, detection.rectangle)
            self.players.append(player)

        # The last frame with tracked data.
        self.last_frame = 0

    def feed_rectangles(self, player_list):
        """ Give the player tracker the next frame's player detection data,
        in the form of a list of detected player rectangles.
        player_list: [((x, y, w, h), team), ...]
        team is the team of the detected rectangle (see enum class below).
        This function will associate the given player rectangles to
        the list of players.

        Current implementation: We'll assume nothing on the number of
        detections given; overdetections or underdetections can have.
        Thus we won't try to solve this as a matching problem. Instead,
        we'll use heuristics to detect collisions and detection drops,
        individually for each player.
        Collisions: when two players are tracked to the same detection.
        Detection fails: when a player cannot be tracked to any nearby
        detection.

        NEW IDEA FOR COLLISIONS:
        We no longer try to deal with collisions separately.
        Instead, we try our best to keep an internal model of the players
        and their underlying locations.
        When a player is matched to its own detection, we say its current
        model is "exact".
        When a player cannot be matched to a detection, we extrapolate
        its location, and say its current model is "extrapolated".
        When a player is matched to a contested detection, it is similar to
        extrapolation, but we use the contested detection (collision) to guide
        the extrapolation. Additionally, we try to aggressively assign nearby
        unclaimed detections to players in the collision (although this is
        only necessary when the player moves far from their extrapolated
        position during the collision).
        """

        # Process player_list into structured data.
        detections = [Detection(x[0], x[1]) for x in player_list]
        self.detections_by_frame.append(detections)

        # -- new version --




        # -- old version --

        # Map each player to some point in the detection data,
        # or handle detection failures appropriately.

        for player in self.players:
            last_frame_pos = player.centroids[self.last_frame]
            centroid_distances = [
                euclidean_distance(last_frame_pos, dtec.centroid)
                for dtec in detections]
            # We define the next location of the player to be:
            # - The nearest detection centroid of the player's color,
            #   within the search radius. If none,
            # - The nearest detection centroid of any color,
            #   within the COLLISION radius. If none,
            # - Detection failed, and we extrapolate their location.
            team_centroid_dists = centroid_distances[:]
            for i in range(len(detections)):
                if player_list[i][1] != player.color:
                    team_centroid_dists[i] = 999999

            argmin = np.array(team_centroid_dists).argmin()
            if team_centroid_dists[argmin] > player.search_radius:
                argmin = None

            if argmin is None:
                # Now try any color.
                # Should use collision radius here.
                argmin = np.array(centroid_distances).argmin()
                if centroid_distances[argmin] > player.search_radius:
                    argmin = None

            if argmin is None:
                # Now just extrapolate.
                player.extrapolate_next_location()
            else:
                # Register the tracking, and update stats for this detection.
                detections[argmin].claimers[player] = None

        # -- Attempt to resolve collisions in this frame. --
        list_of_collisions = [
            (i, detection) for i, detection in enumerate(detections)
            if len(detection.claimers) > 1
        ]
        list_of_unclaimed = [
            (i, detection) for i, detection in enumerate(detections)
            if len(detection.claimers) == 0
        ]
        # For each unclaimed detection, try to match it to a colliding player:
        #   - Find the nearest collision.
        #   - Check that it's in a radius (distance from detection centroid to
        #     collision rectangle outer edge).
        #   - Check that this color is the color of a player in the collision.
        #   - Remove the nearest player of this color from the collision.

        for i, unclaimed in list_of_unclaimed:
            for j, collision in list_of_collisions:
                dist = dist_point_to_rect(unclaimed.centroid, collision.rectangle)
                if dist < 50:  # XXX arbitrary. maybe use rect-rect distance???
                    nearest_player = None
                    min_dist = 999999
                    for player in collision.claimers:
                        if player.color != unclaimed.color: continue
                        dist_player = euclidean_distance(
                            player.centroids[self.last_frame],
                            unclaimed.centroid)
                        if dist_player > min_dist: continue
                        min_dist = dist_player
                        nearest_player = player
                    if nearest_player is not None:
                        unclaimed.claimers[nearest_player] = None
                        del collision.claimers[nearest_player]
                        break

        # Now save all detections.
        for detection in detections:
            if len(detection.claimers) > 1:
                print "Collision detected: players %r" % \
                    [player.pid for player in detection.claimers]
            for player in detection.claimers:
                if len(detection.claimers) > 1:
                    player.extrapolate_next_location(detection.rectangle)
                else:
                    player.set_next_location(
                        detection.centroid, detection.rectangle)

        self.last_frame += 1

class Player(object):
    """ Represents a player, their tracked position, and any player detection
    data necessary to continue tracking their position.

    All fields public. Access directly.
    """
    def __init__(self, pid, color):
        # A number to uniquely identify this player.
        # Currently corresponds with its index in the PlayerTracker.
        self.pid = pid

        # The player's team. as a Color.enum.
        self.color = color

        # How many pixels to search for this player's location in the next frame
        # (that is, the centroid of this player's detection rectangle).
        # Implementation note: will have to increase with every frame
        # that the player goes undetected.
        self.search_radius = 50  # Arbitrary defualt value. May need to tweak.

        # The index of the last "finished" frame; the last frame with exact
        # position and tracking data.
        # XXX TODO: at the end of the video, some players might be
        # in extrapolating mode; we should set their locations.
        self.last_finished = -1

        # XXX: Maybe move homography, smoothing, and distancing into here?

        # -- Fields that are mandatory for every tracking frame. --
        # (That is, we MUST update these fields with every frame we process,
        # because we need these fields to do tracking in the next frame.)

        # Video position data. We use this to find the player in the next frame.
        # We can also extrapolate this when detection data is unavailable,
        # so we know where to search when detection reappears.
        # Current implementation: centroid of the detection rectangle.
        # (x int, y int).
        self.centroids = []

        # One type of player detection data.
        # A list of the player's detected rectangles in frame order,
        # or None if detection failed for the player in that frame.
        # Rectangles are (x, y, width, height). Top left is 0, 0.
        self.rectangles = []

        # A list of the player's ground positions, on the raw video,
        # in frame order, or None if extrapolated.
        # Must eventually be filled in.
        # Position represented by a tuple (x, y) of ints.
        # This is what we really need tracking to produce. Applying homography
        # to get top-down coordinates is trivial.
        # TODO: decide how to calculate these from raw position data.
        # We probably don't have to do this on the fly.
        self.raw_positions = []

        # -- Fields that are mandatory for every finished frame. --

        # The moving average of the last five centroids. We use the *direction*
        # of the moving average to extrapolate the direction of the player.
        # This is necessary to minimize the effect of noise from outlier
        # player centroids.
        self.centroid_mas = []

    def set_next_location(self, centroid, rectangle):
        """ Provide the next frame's exact location data for this player.
        This means the exact rectangle centroid,
        as well as the player rectangle itself.
        """
        self.centroids.append(centroid)
        self.rectangles.append(rectangle)
        # Compute and append the raw position.
        x, y, w, h = rectangle
        self.raw_positions.append((x + w/2, y + h))
        # If this is the first detection after an extrapolation,
        # correct the extrapolated centroid data,
        # and fill in raw positions and centroid moving averages.
        if self.last_finished < len(self.centroids) - 2:
            self.backfill_unfinished_frames(len(self.centroids) - 2)
        # At this point, everything is as if we have exact location data
        # for all frames up to this one.
        # Compute and append the centroid moving average.
        self.process_moving_average(len(self.centroids) - 1)
        self._reset_search_radius()

        self.last_finished += 1

    def extrapolate_next_location(self, bounds=None):
        """ Extrapolate and set the player location data for the next frame.
        This is just calculating and saving the predicted centroid.
        bounds: a rectangle that the next location should be in. Used during
        collisions.
        """
        # Implementation: We use the velocity of the last exact moving average.
        if self.last_finished == 0:
            extrapolated_velocity = np.array((0, 0))
        else:
            extrapolated_velocity = \
                np.array(self.centroid_mas[self.last_finished]) - \
                np.array(self.centroid_mas[self.last_finished - 1])

        # XXX: and search radius too? prolly dun need

        last_position = np.array(self.centroids[-1])
        extrapolated_position = tuple(last_position + extrapolated_velocity)
        # If the extrapolated position is outside of the bounds rectangle,
        # move it to the nearest location within the rectangle.
        if bounds is not None:
            x, y, w, h = bounds
            extrapolated_position = (
                max(extrapolated_position[0], x),
                max(extrapolated_position[1], y),
            )
            extrapolated_position = (
                min(extrapolated_position[0], x + w),
                min(extrapolated_position[1], y + h),
            )

        self.centroids.append(extrapolated_position)
        self.rectangles.append(None)
        self.raw_positions.append(None)
        self._increment_search_radius()

    def backfill_unfinished_frames(self, fno):
        """ Backfills unfinished frames up to fno, inclusive, if any,
        using exact location data from frame fno+1 (the current frame).
            - Readjusts self.centroids for the duration of the extrapolation.
            - Fills in self.raw_positions.
            - Fills in self.centroid_mas.
        """

        # Readjusting self.centroids, and fill in raw positions.
        # Basically, just draw a line from the
        # last exact detection to this exact detection.
        num_frames_to_fill = (fno + 1) - self.last_finished

        initial_centroid = np.array(self.centroids[self.last_finished])
        initial_rp = np.array(self.raw_positions[self.last_finished])

        final_centroid = np.array(self.centroids[fno + 1])
        final_rp = np.array(self.raw_positions[fno + 1])

        centroid_displacement = np.array(final_centroid - initial_centroid)
        rp_displacement = np.array(final_rp - initial_rp)
        # Fill in self.centroids from the frame after last_finished to
        # fno, inclusive.
        for i, k in enumerate(range(self.last_finished, fno + 1)):
            if i == 0: continue  # The last finished frame, already finished.
            assert num_frames_to_fill > 0
            assert i < num_frames_to_fill
            adjusted_centroid = initial_centroid + \
                ((i * centroid_displacement) / num_frames_to_fill)
            adjusted_rp = initial_rp + \
                ((i * rp_displacement) / num_frames_to_fill)
            self.centroids[k] = tuple(adjusted_centroid)
            self.raw_positions[k] = tuple(adjusted_rp)

        # Fill in the centroid moving averages, up to fno.
        for k in range(self.last_finished + 1, fno + 1):
            self.process_moving_average(k)
            self.last_finished += 1


    def process_moving_average(self, fno):
        """ Compute the moving average for fno,
        using data from frames fno up to fno-4, and save it into the player.
        """
        # Make sure frames before fno are complete.
        assert len(self.centroid_mas) - 1 == self.last_finished
        assert len(self.centroid_mas) == fno, \
            "Last finished frame was %r " + \
            "but we are computing moving average for %r" % \
            (len(self.centroid_mas) - 1, fno)
        centroid_ma = np.array((0, 0))
        k = 0
        while k < 5:
            i = fno - k
            if i < 0: break
            centroid_ma += np.array(self.centroids[i])
            k += 1
        assert k > 0
        centroid_ma /= k
        self.centroid_mas.append(centroid_ma)

    def _reset_search_radius(self):
        self.search_radius = 50

    def _increment_search_radius(self):
        self.search_radius += 3

class Detection(object):
    """ Represents a single raw detection datum for some frame.
    All fields public; access directly.
    """
    def __init__(self, rectangle, color):
        """ rectangle: (x int, y int, w int, h int)
            color: a Color enum
        """
        # hmm: do we need to store a key too?
        self.rectangle = rectangle
        self.color = color
        self.centroid = rect_centroid(self.rectangle)
        # The list of players who have claimed this
        # detection as their next location.
        # A hashset: { player: None }
        self.claimers = {}

if __name__ == '__main__':
    test()