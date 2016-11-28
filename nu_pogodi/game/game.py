# coding=utf-8
import random
import time

import cv2

from nu_pogodi import utils
from nu_pogodi.game.player import Player
from nu_pogodi.images import read_image


class EggIsAlreadyAdded(Exception):
    pass


class EggHill(object):
    class PositionStatus:
        EMPTY = 0
        EGG = 1

    def __init__(self):
        self.positions = [self.PositionStatus.EMPTY] * 5
        self.last_egg_was_lost = False

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
        last_egg = self.positions.pop(-1)
        self.last_egg_was_lost = last_egg == self.PositionStatus.EGG


class Game(object):
    PLAYER_AREA_START_POSITION = 280, 260
    PLAYER_AREA_SIZE = 300, 414

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
        self._bg = read_image('field.jpg')
        self._basket_img = read_image('basket.png')
        self._last_hill_update = 0
        self.egg_hills = {
            k: EggHill()
            for k in ['left_upper', 'left_lower',
                      'right_upper', 'right_lower']
        }
        self._egg_speed = 1
        self._player = Player()
        self._player_score = 0

        self._initial_frame_size = None

    def initialize(self, first_frame):
        self._initial_frame_size = first_frame.shape[:2]
        self._player.initialize(first_frame)

    def show(self, img):
        game_frame = self._get_game_frame(img)

        self._update_state()
        self._player.update(img)

        for hill_key, hill in self.egg_hills.iteritems():
            if hill.egg_on_position(-1):
                if self._does_player_can_catch_egg(
                        self.EGG_POSITIONS[hill_key][-1]):
                    hill.rm_last()
                    self._player_score += 1

        self._show_eggs(game_frame)
        self._show_player_hands(game_frame)
        self._show_score(game_frame)

        return game_frame

    def _show_score(self, game_frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(game_frame,
                    str(self._player_score),
                    (550, 150),
                    font,
                    fontScale=1.5,
                    color=(0, 0, 0),
                    thickness=4)

    def _does_player_can_catch_egg(self, egg_position):
        return any(
            utils.get_points_distance(
                self._get_hand_position(hand.position), egg_position) < 40
            for hand in [self._player.right_hand, self._player.left_hand]
        )

    def _update_state(self):
        cur_time = time.time()
        if cur_time - self._last_hill_update > self._egg_speed:
            self._last_hill_update = cur_time
            for hill in self.egg_hills.values():
                hill.update()
                if hill.last_egg_was_lost:
                    self._player_score = max(0, self._player_score - 1)

            hills = self.egg_hills.values()
            random.shuffle(hills)
            for hill in hills:
                try:
                    hill.add_egg()
                    break
                except EggIsAlreadyAdded:
                    pass

    def _get_game_frame(self, img):
        x, y = self.PLAYER_AREA_START_POSITION
        h, w = self.PLAYER_AREA_SIZE
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

    def _get_hand_position(self, hand_position):
        return (
            hand_position[0] * self.PLAYER_AREA_SIZE[0] /
                    self._initial_frame_size[0] +
                self.PLAYER_AREA_START_POSITION[0],
            hand_position[1] * self.PLAYER_AREA_SIZE[1] /
                    self._initial_frame_size[1] +
                self.PLAYER_AREA_START_POSITION[1],
        )

    def _show_player_hands(self, game_frame):
        for hand in [self._player.left_hand,
                     self._player.right_hand]:
            if hand.position is not None:
                hand_position = self._get_hand_position(hand.position)
                basket_position = (
                    hand_position[0] - self._basket_img.shape[1] / 2,
                    hand_position[1] - int(self._basket_img.shape[0] / 2),
                )
                utils.insert_picture(
                    game_frame, self._basket_img, basket_position)