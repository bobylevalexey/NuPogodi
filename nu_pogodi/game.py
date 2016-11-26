import os
import random
import time

import cv2

from nu_pogodi import utils


class EggIsAlreadyAdded(Exception):
    pass


class EggHill(object):
    class PositionStatus:
        EMPTY = 0
        EGG = 1

    def __init__(self):
        self.positions = [self.PositionStatus.EMPTY] * 5

    def rm_last(self):
        self.positions[-1] = self.PositionStatus.EMPTY

    def egg_on_position(self, position):
        return self.positions[position] == self.PositionStatus.EGG

    def add_egg(self):
        if self.egg_on_position(0):
            raise EggIsAlreadyAdded
        self.positions[0] = self.PositionStatus.EGG

    def update(self):
        self.positions.insert(0, self.PositionStatus.EMPTY)
        self.positions.pop(-1)


# class Hand(object):
#     def __init__(self):
#         self._prev_position = None
#         self.position =


class Player(object):
    def __init__(self):
        self.left_hand_position = None
        self.right_hand_position = None

        self._prev_gray = None

    def update(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self._prev_gray = gray
        mv_channel = utils.get_moves_channel(flow)

        contours = list(utils.iter_outer_contours(mv_channel, min_area=1000 ))
        if contours:
            _, left_hand_candidate = max(
                ((c, max(c, key=utils.get_point_y)) for c in contours),
                key=lambda x: utils.get_point_y(x[1])
            )
            _, right_hand_candidate = min(
                ((c, min(c, key=utils.get_point_y)) for c in contours),
                key=lambda x: utils.get_point_y(x[1])
            )

            self.left_hand_position = tuple(*left_hand_candidate)
            self.right_hand_position = tuple(*right_hand_candidate)

    def can_catch(self, position):
        return any(utils.get_points_distance(h, position) < 20
                   for h in (self.left_hand_position,
                             self.right_hand_position))


class Game(object):
    IMAGES_DIR = os.path.join(
        os.path.dirname(__file__),
        'images'
    )
    EGG_COLOR = (70, 70, 70)

    class GetPositionsFunc:
        @staticmethod
        def left(x0, y0):
            return [
                (x0, y0),
                (x0 + 30, y0 + 14),
                (x0 + 60, y0 + 28),
                (x0 + 90, y0 + 44),
                (x0 + 119, y0 + 79),
            ]

        @staticmethod
        def right(x0, y0):
            return [
                (x0, y0),
                (x0 - 30, y0 + 14),
                (x0 - 60, y0 + 28),
                (x0 - 90, y0 + 44),
                (x0 - 119, y0 + 79),
            ]

    EGG_POSITIONS = {
        'left_upper': GetPositionsFunc.left(161, 263),
        'left_lower': GetPositionsFunc.left(161, 390),
        'right_upper': GetPositionsFunc.right(780, 263),
        'right_lower': GetPositionsFunc.right(780, 390),
    }

    def __init__(self):
        self._bg = self._get_image('field.jpg')
        self._last_hill_update = 0
        self.egg_hills = {
            k: EggHill()
            for k in ['left_upper', 'left_lower',
                      'right_upper', 'right_lower']
        }
        self._egg_speed = 3
        self._player = Player()
        self._player_score = 0

    def show(self, img):
        game_frame = self._get_game_frame(img)

        self._update_state()
        self._player.update(game_frame)

        for hill_key, hill in self.egg_hills.iteritems():
            if hill.egg_on_position(-1):
                if self._player.can_catch(self.EGG_POSITIONS[hill_key][-1]):
                    hill.rm_last()
                    self._player_score += 1
                    print self._player_score

        self._show_eggs(game_frame)
        self._show_player_hands(game_frame)

        return game_frame

    def _update_state(self):
        cur_time = time.time()
        if cur_time - self._last_hill_update > self._egg_speed:
            self._last_hill_update = cur_time
            for hill in self.egg_hills.values():
                hill.update()

            hills = self.egg_hills.values()
            random.shuffle(hills)
            for hill in hills:
                try:
                    hill.add_egg()
                    break
                except EggIsAlreadyAdded:
                    pass

    def _get_game_frame(self, img):
        x, y = 280, 260
        h, w = 300, 415
        game_frame = self._bg.copy()
        game_frame[x: x + h, y: y + w] = cv2.resize(img, (w, h))
        return game_frame

    def _show_eggs(self, game_frame):
        for hill_key in self.egg_hills:
            self._show_hill_eggs(
                game_frame, self.egg_hills[hill_key],
                self.EGG_POSITIONS[hill_key])

    def _show_hill_eggs(self, img, hill, show_positions):
        for pos, position_point in enumerate(show_positions):
            if hill.egg_on_position(pos):
                cv2.circle(img, position_point, 12, self.EGG_COLOR, 4)

    def _get_image(self, file_name):
        return cv2.imread(os.path.join(self.IMAGES_DIR, file_name))

    def _show_player_hands(self, game_frame):
        if self._player.left_hand_position is not None:
            cv2.circle(
                game_frame, self._player.left_hand_position,
                5, (255, 0, 0), 3)
        if self._player.right_hand_position is not None:
            cv2.circle(
                game_frame, self._player.right_hand_position,
                5, (0, 255, 0), 3)
