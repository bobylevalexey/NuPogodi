import cv2

from nu_pogodi.game import Game
from nu_pogodi import utils

if __name__ == "__main__":
    game = Game()

    frames = utils.iter_cam_frames()

    game.initialize(next(frames))

    for frame in frames:
        cv2.imshow('Video', game.show(frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
