import cv2

from nu_pogodi import utils


def get_most_left_point(contour):
    left_point = min(contour, key=lambda x: x[0][0])
    return tuple(left_point[0])


def get_most_right_point(contour):
    left_point = max(contour, key=lambda x: x[0][0])
    return tuple(left_point[0])


class BaseHand(object):
    def __init__(self):
        self.position = None

    def update(self, mv_channel):
        new_position = self._find_new_position_candidate(mv_channel)
        if new_position:
            self.position = new_position

    def _find_new_position_candidate(self, mv_channel):
        hulls = []
        for cnt in utils.iter_outer_contours(self._get_search_area(mv_channel)):
            hull = cv2.convexHull(cnt)
            if cv2.contourArea(hull) < 3000:
                continue
            hulls.append(hull)

        if hulls:
            return self._get_position(hulls)

    def _get_search_area(self, mv_channel):
        raise NotImplemented

    def _get_position(self, hulls):
        raise NotImplemented


class RightHand(BaseHand):
    def _get_position(self, hulls):
        position, _ = max(((get_most_right_point(h), h) for h in hulls),
                          key=lambda x: x[0][0])
        return position

    def _get_search_area(self, mv_channel):
        return mv_channel[:, mv_channel.shape[1] / 2:]

    def _find_new_position_candidate(self, mv_channel):
        new_position = \
            super(RightHand, self)._find_new_position_candidate(mv_channel)
        if new_position:
            return (new_position[0] + mv_channel.shape[1] / 2,
                    new_position[1])


class LeftHand(BaseHand):
    def _get_position(self, hulls):
        position, _ = min(((get_most_left_point(h), h) for h in hulls),
                          key=lambda x: x[0][0])
        return position

    def _get_search_area(self, mv_channel):
        return mv_channel[:, :mv_channel.shape[1] / 2]


class Player(object):
    def __init__(self):
        self.left_hand = LeftHand()
        self.right_hand = RightHand()
        self._vertical_center = None

        self._prev_gray = None

    def _get_gray_frame(self, frame):
        return cv2.pyrDown(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def initialize(self, first_frame):
        self._prev_gray = self._get_gray_frame(first_frame)
        self._horizontal_middle = first_frame.shape[1] / 2

    def _get_mv_channel(self, gray_channel):
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray_channel, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return utils.get_moves_channel(flow)

    def update(self, frame):
        gray = self._get_gray_frame(frame)
        mv_channel = cv2.pyrUp(self._get_mv_channel(gray))
        self._prev_gray = gray

        self.left_hand.update(mv_channel)
        self.right_hand.update(mv_channel)

    def can_catch(self, position):
        return any(utils.get_points_distance(h, position) < 20
                   for h in (self.left_hand.position,
                             self.right_hand.position))
