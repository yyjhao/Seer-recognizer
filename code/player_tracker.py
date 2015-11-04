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

# Add bounding to player tracking during collisions.
# Fix extrapolation (should be some sort of direction of moving average,
# which we will have to start recording)
# Implement collision -> (collision, collision) resolution as described
# in comments.
# Manually check and tweak search radiuses (add imcrementing if necessary).
# After that, it should actually be all good...


import cv2
import numpy as np

from player_detector import (
    Color,
    getPlayers,
)
from util import (
    euclidean_distance,
    rect_centroid,
)

def test():
    cap = cv2.VideoCapture('../videos/stitched.mpeg')
    NUM_SKIP = 2
    for i in range(NUM_SKIP):
        _, _ = cap.read()
    _, frame = cap.read()
    players = getPlayers(frame)
    pt = PlayerTracker(players)
    print "Processed frame 0."
    i = 1
    NUM_FRAMES = 2
    while i < NUM_FRAMES:
        _, frame = cap.read()
        rects = getPlayers(frame)
        pt.feed_rectangles(rects)
        print "Processed frame %r." % i
        i += 1

    del cap

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
                loc = (rect[0] + rect[2] / 2, rect[1] + rect[3])
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
        # The list of collisions in the last frame processed.
        self.collisions = []

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

        Dealing with collisions:
        A frame has a collision if multiple of its players are mapped
        to the same centroid coordinate.
        During frame processing, if the last frame had a collision,
        we must do additional checks to see if the collision can be resolved
        (otherwise the two players are stuck together forever).
        Our current heuristic for this is to check for any unclaimed
        detections `around` the collision (except the collision's new
        position). If so, we `select` a player to remove from the collision
        and associate with the new point instead.
        By `around`, we mean near the size of the collision rectangle
        (although tweak can).
        By `select`, we mean heuristically; the player must be the same
        colour, and if still ambiguous, it should be the player with the
        closest expected position.
        To facilitate all of this, during a collision, we must record the
        player's expected position instead of his collision centroid,
        and optionally, after a collision, retroactively update his
        positions during the collision.
        """

        # Process player_list into structured data.
        detections = [Detection(x[0], x[1]) for x in player_list]
        self.detections_by_frame.append(detections)

        # Map each player to some point in the detection data,
        # or handle detection failures appropriately.
        # NB: Only do this for players not involved in a collision.
        # For players currently in collisions, we will track/resolve
        # them separately.

        # Players "exactly detected" in the following code might actually
        # be colliding (obv we don't know until we process everybody),
        # so we use this dict to tentatively register detections.
        # { Player: detection index }
        tentative_exact_pos = {}

        for player in self.players:
            # Skip player if they're in a collision.
            if not all(
                    player not in collision.players
                    for collision in self.collisions):
                continue
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
                # We should probably move this into the player class.
                player.extrapolate_next_location()
            else:
                # Register the tracking, and update stats for this detection.
                tentative_exact_pos[player] = argmin
                detections[argmin].claimers.append(player)

        # -- Attempt to resolve collisions from the last frame. --
        collision_index = 0
        collisions_to_delete = []
        for collision in self.collisions:
            # Find where this collision is going.
            last_frame_pos = collision.centroid
            centroid_distances = [
                euclidean_distance(last_frame_pos, dtec.centroid)
                for dtec in detections]
            argmin = np.array(centroid_distances).argmin()

            # See if we can resolve any players to nearby detections.
            unclaimed_rectangles = [
                player_list[i] if len(detections[i].claimers) == 0 else None
                for i in range(len(player_list))]
            unclaimed_rectangles[argmin] = None
            resolutions = collision.try_resolving(
                unclaimed_rectangles, self.last_frame)
            # Register trackings. We can definitely refactor this.
            for index_centroid, index_player in resolutions:
                centroid = detections[index_centroid].centroid
                player = collision.players[index_player]
                tentative_exact_pos[player] = index_centroid
                detections[index_centroid].claimers.append(player)
                # Also, remove this player from the collision.
                del collision.players[index_player]
                print "Removing player %r from a collision." % player.pid
            assert len(collision.players) > 0, \
                "Collision objects should always have a last player!"
            # Remove this collision if it is fully resolved.
            if len(collision.players) == 1:
                print "Collision fully resolved."
                player = collision.players[0]
                tentative_exact_pos[player] = argmin
                detections[argmin].claimers.append(player)
                del collision.players[0]
                collisions_to_delete.append(collision_index)
            # Otherwise, reprocess the collisions for this frame.
            else:
                for player in collision.players:
                    detections[argmin].claimers.append(player)

        # All the collision data has been processed into resolutions or
        # new collisions. We can now reset the collisions.
        self.collisions = []

        # Add all detections with multiple claims as collisions.
        # TODO: Deal with new players joining a collision.
        # XXX: Persist or redetect existing collisions? Redetect sounds easier
        for i in range(len(detections)):
            if len(detections[i].claimers) > 1:
                collision = Collision(
                    detections[i].claimers, detections[i].centroid, player_list[i][0])
                self.collisions.append(collision)
        # Track the expected position of players involved in collisions.
        # These players may have tentative exact locations, so get rid of those too.
        for collision in self.collisions:
            for player in collision.players:
                if player in tentative_exact_pos:
                    del tentative_exact_pos[player]
                player.extrapolate_next_location()

        # Now, all players in tentative_exact_pos are definitely not in collisions,
        # so they are no longer tentative -- let's save them!
        for player, index in tentative_exact_pos.iteritems():
            player.set_next_location(
                detections[index].centroid, detections[index].rectangle)

        self.last_frame += 1


class Collision(object):
    """ XXX: Represents a collision (in a given frame ???) """
    # XXX: yes, collisions are all per-frame independent.
    # remember: a collision can resolve into two collisions!!
    # todo: implement this. will have to include projected collision destination
    # in the search, and make sure at least one player goes there (so resolve player
    # cloest to collision dest first). and then resolve all other players.
    #
    def __init__(self, players, centroid, rectangle):
        """ players: The list of Player objects involved in the collision.
            centroid: The centroid of the collision.
            rectangle: The rectangle of the collision.
        """
        self.players = players
        self.centroid = centroid
        self.rectangle = rectangle
        print "Collision detected at %r. Players: %r" % (centroid, [p.pid for p in self.players])

    def try_resolving(self, list_of_detections, last_frame):
        """ Given a list of ((rectangle, color) or None),
        figure out if any of the players in this collision could have
        ended up at these detections.
        Must always keep one player in this collision
        (since the collision is tracked).
        Returns a list of (index_centroid, index_player), where
        index_centroid is the index of this centroid in the original
        list_of_points, and index_player is the index of the matching player in
        this collision.
        """
        search_radius = (self.rectangle[2] + self.rectangle[3])  # Arbitrary.
        list_of_centroids = [
            (rect_centroid(x[0]), x[1])
            if x is not None else None
            for x in list_of_detections
        ]
        list_of_resolutions = []
        # {player index: (index of closest centroid, dist to centroid)}
        list_of_candidates = {}
        for i in range(len(self.players)):
            player = self.players[i]
            dist_to_centroids = [
                euclidean_distance(player.centroids[last_frame], x[0])
                if x is not None and player.color == x[1] else 999999
                for x in list_of_centroids
            ]
            argmin = np.array(dist_to_centroids).argmin()
            list_of_candidates[i] = (argmin, dist_to_centroids[argmin])

        def candidate_argmin_key(list_of_candidates):
            assert len(list_of_candidates) > 0, \
                "Cannot argmin an empty dict."
            items = [x for x in list_of_candidates.iteritems()]
            argmin = list_of_candidates.keys()[0]
            minval = items[0][1][1]
            for x in items:
                if x[1][1] < minval:
                    argmin = x[0]
            return argmin

        best_match_player_key = candidate_argmin_key(list_of_candidates)
        best_match_player = list_of_candidates[best_match_player_key]
        if best_match_player[1] < search_radius:
            list_of_resolutions.append((
                best_match_player[0], best_match_player_key))
            list_of_centroids[best_match_player[0]] = None
            del list_of_candidates[best_match_player_key]
        else:
            return []
        while len(list_of_candidates) > 1:
            for i in list_of_candidates:
                player = self.players[i]
                dist_to_centroids = [
                    euclidean_distance(player.centroids[last_frame], x[0])
                    if x is not None and player.color == x[1] else 999999
                    for x in list_of_centroids
                ]
                argmin = np.array(dist_to_centroids).argmin()
                list_of_candidates[i] = (argmin, dist_to_centroids[argmin])
            best_match_player_key = candidate_argmin_key(list_of_candidates)
            best_match_player = list_of_candidates[best_match_player_key]
            if best_match_player[1] < search_radius:
                list_of_resolutions.append((
                    best_match_player[0], best_match_player_key))
                list_of_centroids[best_match_player[0]] = None
                del list_of_candidates[best_match_player_key]
            else:
                break

        return list_of_resolutions

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

        # XXX: Maybe move homography, smoothing, and distancing into here?

        # -- Fields that are mandatory for every frame. --

        # A list of the player's ground positions, on the raw video,
        # in frame order.
        # Position represented by a tuple (x, y) of ints.
        # This is what we really need tracking to produce. Applying homography
        # to get top-down coordinates is trivial.
        # TODO: decide how to calculate these from raw position data.
        # We probably don't have to do this on the fly.
        self.raw_positions = []

        # Video position data. We use this to find the player in the next frame.
        # We can also extrapolate this when detection data is unavailable,
        # so we know where to search when detection reappears.
        # Current implementation: centroid of the detection rectangle.
        # (x int, y int).
        self.centroids = []

        # -- Fields that are optional for every frame (might be None). --

        # One type of player detection data.
        # A list of the player's detected rectangles in frame order,
        # or None if detection failed for the player in that frame.
        # Rectangles are (x, y, width, height). Top left is 0, 0.
        self.rectangles = []

    def set_next_location(self, centroid, rectangle):
        """ Provide the next frame's exact location data for this player.
        This means the exact rectangle centroid,
        as well as the player rectangle itself.
        """
        self.centroids.append(centroid)
        self.rectangles.append(rectangle)
        # Set self.raw_positions as well, backfilling undetections if any.
        # And adjust backfilled (zipping) centroids as well?

    def extrapolate_next_location(self):
        """ Extrapolate and set the player location data for the next frame.
        This is just calculating and setting the predicted centroid.
        """
        # XXX: a bounds parameter. in a collision, we KNOW the player
        # is within the collision box.
        # XXX: and search radius too? prolly dun need

        # Currently, we'll do this by doing a flat average
        # of velocity data from the past [up to] five centroids
        # (four velocities).
        extrapolated_velocity = np.array((0, 0))
        samples = 0
        last_frame = len(self.centroids) - 1
        # XXX im retarded, fix this. this is no bueno at all
        f2 = last_frame
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
        self.rectangles.append(None)
        # self.raw_positions?

    # XXX i have no idea. currently disabled
    def reset_search_radius(self):
        # The search radius increases for every failed detection frame.
        # It is reset when detection succeeds again.
        # This currently faciliates a public reset,
        # although we should probably write handlers in this class
        # to trigger all of this stuff.
        self.search_radius = 50
    # XXX i ahve no idea
    def increment_search_radius(self):
        self.search_radius += 10

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
        self.claimers = []

if __name__ == '__main__':
    test()