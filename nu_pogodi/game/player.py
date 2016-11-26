import cv2

from nu_pogodi import utils


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
