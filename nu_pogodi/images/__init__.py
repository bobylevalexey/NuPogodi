import os

import cv2


_IMAGES_DIR = os.path.dirname(__file__)


def read_image(image_file_name, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imread(os.path.join(_IMAGES_DIR, image_file_name), flags)
