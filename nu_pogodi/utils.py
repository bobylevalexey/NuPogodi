# coding=utf-8
import cv2
import math
import numpy
import numpy as np


def iter_cam_frames():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        yield cv2.flip(frame, 1)

    cap.release()
    cv2.destroyAllWindows()


def threshold_frame(frame, param=127):
    """
    пиксели меньшие param устанавливает нулями. в остальные устанвливает 255
    """
    ret,thresh = cv2.threshold(frame,param,255,cv2.THRESH_BINARY)
    return thresh


def morphology_transform(frame, kernel=5):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    kernel = np.ones(kernel, np.uint8)
    return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)


def glue_pictures(*images):
    return numpy.hstack(images)


def to_multi_channel(channel):
    return cv2.merge((channel, channel, channel))


def smooth_image(image, smooth_param):
    return cv2.blur(image, (10, 10))


def iter_outer_contours(channel, min_area=None):
    _, contours, heirs = cv2.findContours(
        channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try:
        heirs = heirs[0]
    except:
        heirs = []

    for cnt, heir in zip(contours, heirs):
        if min_area is not None:
            if cv2.contourArea(cnt) < min_area:
                continue
        _, _, _, outer_i = heir
        if outer_i < 0:
            yield cnt


def get_moves_channel(flow, mv_param=1):
    fx, fy = flow[:,:,0], flow[:,:,1]
    moves = np.sqrt(fx*fx+fy*fy).astype(np.uint8)
    fr = threshold_frame(moves, param=mv_param)
    return fr


def get_points_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in xrange(2)))


def insert_picture(frame, pic, offset):
    x_offset, y_offset = offset
    pic_h, pic_w = pic.shape[:2]
    pic_alpha_channel = pic[:, :, 3]

    for channel in range(3):
        old_frame_area = frame[y_offset: y_offset + pic_h,
                               x_offset: x_offset + pic_w,
                               channel]
        new_frame_area = pic[:, :, channel] * (pic_alpha_channel / 255.0) + \
                         old_frame_area * (1.0 - pic_alpha_channel / 255.0)
        frame[y_offset: y_offset + pic_h,
              x_offset: x_offset + pic_w,
              channel] = new_frame_area
