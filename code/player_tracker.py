#!/usr/bin/env python

# Implementation of player tracking for final project.
# CS 4243, Computer Vision and Pattern Recognition, November 2015.
# Xiayue (Charles) Lin, A0145415L

import cv2
import numpy as np

from player_detector import (
    Color,
    getPlayers,
)

def test():
    cap = cv2.VideoCapture('../videos/stitched.mpeg')
    _, frame = cap.read()
    players = getPlayers(frame)
    pt = PlayerTracker(players)
    i = 1
    while i < 50:
        _, frame = cap.read()
        rects = getPlayers(frame)
        pt.feed_rectangles(rects)
        print "Processed frame %r." % i
        i += 1

    del cap

    cap = cv2.VideoCapture('../videos/stitched.mpeg')
    for i in range(50):
        _, frame = cap.read()
        pno = 0
        print pt.players
        for player in pt.players:
            rect = player.detection_rectangles[i]
            if rect is not None:
                loc = (rect[0] + rect[2] / 2, rect[1] + rect[3])
                print loc
                cv2.circle(frame, loc, 5, player.color, -1)
                cv2.putText(frame, str(pno), loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
            else:
                loc = player.centroids[i]
                print "%r (extrapolated)" % (loc,)
                cv2.circle(frame, loc, 5, (255, 255, 255), -1)
                cv2.putText(frame, str(pno), loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
            pno += 1
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
        # The players being tracked, as a list of Player objects.
        self.players = []
        i = 0
        for player_data in player_list:
            player = Player(i)
            player.color = player_data[1]
            player.detection_rectangles.append(player_data[0])
            player.centroids.append(rect_centroid(player_data[0]))
            self.players.append(player)
            i += 1
        # The last frame with tracked data.
        self.last_frame = 0

    def feed_rectangles(self, player_list):
        """ Give the player tracker the next frame's player detection data,
        in the form of a list of detected player rectangles.
        player_list: [((x, y, w, h), team), ...]
        team is the team of the detected rectangle (see enum class below).
        This function will associate the given player rectangles to
        the list of players.
        XXX edge case: collision
        XXX edge case: detection failures
        """
        # Idea: We'll assume nothing on the number of detections given;
        # can have overdetections or underdetections.
        # Thus we won't try to solve this as a matching problem. Instead,
        # we'll use heuristics to detect collisions and detection drops,
        # individually for each player.
        # Collisions: when two players are tracked to the same detection.
        # Detection fails: when a player cannot be tracked to any nearby
        # detection.

        # Ideally, we'd use a data structure to hold processed data
        # and stats for each detection, but let's be lazy.
        # https://www.youtube.com/watch?v=Nn5K-NNmgTM
        centroids = [rect_centroid(x[0]) for x in player_list]
        # The players that have claimed this detection as their
        # next location.
        detections = [[] for _ in player_list]

        # Map each player to some point in the detection data,
        # or handle detection failures appropriately.
        for player in self.players:
            last_frame_pos = player.centroids[self.last_frame]
            centroid_distances = [
                euclidean_distance(last_frame_pos, x) 
                for x in centroids]
            # We define the next location of the player to be:
            # - The nearest detection centroid of the player's color,
            #   within the search radius. If none,
            # - The nearest detection centroid of any color,
            #   within the search radius. If none,
            # - Detection failed, and we extrapolate for the
            #   next search position.
            # The following code implements this (although written in a 
            # slightly different order).
            argmin = np.array(centroid_distances).argmin()
            if player_list[argmin][1] != player.color:
                # Invalidate all distances to centroids of other colors.
                for i in range(len(centroids)):
                    if centroids[i][1] != player.color:
                        centroid_distances[i] = 999999
                # Now get the closest centroid -- of the player's color.
                argmin_team = np.array(centroid_distances).argmin()
                # If it's in the search radius, use it.
                if centroid_distances[argmin_team] <= player.search_radius:
                    argmin = argmin_team
            if centroid_distances[argmin] > player.search_radius:
                argmin = None

            if argmin is None:
                # We should probably move this into the player class.
                player.extrapolate_centroid(self.last_frame)
                player.detection_rectangles.append(None)
                player.increment_search_radius()
            else:
                # Register the tracking, and update stats for this detection.
                player.detection_rectangles.append(player_list[argmin][0])
                player.centroids.append(centroids[argmin])
                detections[argmin].append(player)
                player.reset_search_radius()

        # TODO: PROCESS COLLISIONS.

        self.last_frame += 1




class Player(object):
    """ Represents a player, their tracked position, and any player detection
    data necessary to continue tracking their position.

    All fields public. Access directly.
    """
    def __init__(self, pid):
        # A number to uniquely identify this player.
        # Currently corresponds with its index in the PlayerTracker.
        self.pid = pid

        # The player's team. as a Color.enum.
        self.color = None

        # A list of the player's ground positions, on the raw video, 
        # in frame order.
        # Position represented by a tuple (x, y) of ints.
        # This is what we really need tracking to produce. Applying homography
        # to get top-down coordinates is trivial.
        # TODO: decide how to calculate these from raw position data. 
        # We probably don't have to do this on the fly.
        self.raw_positions = []

        # Raw positions, but smoothed.
        # XXX may not be necessary.
        # Also, this may look better if done post-homography.
        self.smoothed_positions = []

        # One type of player detection data.
        # A list of the player's detected rectangles in frame order,
        # or None if detection failed for the player in that frame.
        # Rectangles are (x, y, width, height). Top left is 0, 0.
        # XXX: we won't have rectangles for all frames.
        # XXX: maybe we don't need to keep all historical rectangles?
        self.detection_rectangles = []

        # Video position data. We use this to find the player in the next frame.
        # We can also extrapolate this when detection data is unavailable,
        # so we know where to search when detection reappears.
        # Current implementation: centroid of the detection rectangle.
        # (x int, y int).
        self.centroids = []

        # How many pixels to search for this player's location in the next frame
        # (that is, the centroid of this player's detection rectangle).
        # Implementation note: will have to increase with every frame
        # that the player goes undetected.
        self.search_radius = 100  # Arbitrary defualt value. May need to tweak.

    def extrapolate_centroid(self, last_frame):
        """Approximate the centroid point for the next frame."""
        # Currently, we'll do this by doing a flat average
        # of velocity data from the past [up to] five centroids
        # (four velocities). 
        extrapolated_velocity = np.array((0, 0))
        samples = 0
        f2 = last_frame
        # XXX im retarded, fix this
        while samples < 5:
            f1 = f2 - 1
            if f1 < 0: break
            sample_velocity = np.array((
                self.centroids[f2][0] - self.centroids[f1][0],
                self.centroids[f2][1] - self.centroids[f1][1],
            ))
            extrapolated_velocity += sample_velocity
            samples += 1
            f2 = f1
        if samples != 0:
            extrapolated_velocity /= samples
        last_pos = np.array(self.centroids[last_frame])
        self.centroids.append(tuple(last_pos + extrapolated_velocity))

    def reset_search_radius(self):
        # The search radius increases for every failed detection frame.
        # It is reset when detection succeeds again.
        # This currently faciliates a public reset,
        # although we should probably write handlers in this class
        # to trigger all of this stuff.
        self.search_radius = 100

    def increment_search_radius(self):
        self.search_radius += 10

def rect_centroid(rect):
    """ Given a rectangle defined by rect = (x, y, w, h),
    return the location of its centroid, as a tuple (x, y).
    """
    x, y, w, h = rect
    return (x + (w / 2), y + (h / 2))

def euclidean_distance(p1, p2):
    """ Computes the Euclidean distance between two points
    p1 = (x1, y1) and p2 = (x2, y2). 
    Returns a float >= 0.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))

if __name__ == '__main__':
    test()