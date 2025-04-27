import numpy as np


def get_eye_centers(landmarks):
    left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
    right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
    return left_eye, right_eye


def calculate_rotation_angle(left_eye, right_eye):
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    # Limit rotation angle
    return min(max(angle, -10), 10)


def calculate_translation(eye_center, canvas_w, canvas_h):
    target_eye_x = canvas_w // 2
    target_eye_y = int((2 / 5) * canvas_h)
    move_x = target_eye_x - eye_center[0]
    move_y = target_eye_y - eye_center[1]
    return move_x, move_y
