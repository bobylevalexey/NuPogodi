import cv2

from nu_pogodi.game import Game
from nu_pogodi import utils

if __name__ == "__main__":
    game = Game()

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        cv2.imshow('Video', game.show(frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
