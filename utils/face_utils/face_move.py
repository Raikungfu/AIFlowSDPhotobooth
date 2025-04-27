import numpy as np
import cv2


def translate_image(image_np, move_x, move_y, canvas_w, canvas_h):
    translation_matrix = np.float32([[1, 0, move_x], [0, 1, move_y]])
    return cv2.warpAffine(image_np, translation_matrix, (canvas_w, canvas_h), borderValue=(255, 255, 255))


def rotate_image(image_np, eye_center, angle):
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    return cv2.warpAffine(image_np, rotation_matrix, (image_np.shape[1], image_np.shape[0]), borderValue=(255, 255, 255))