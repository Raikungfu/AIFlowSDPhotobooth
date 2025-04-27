from models.face_detection import face_detector, landmark_predictor


def detect_faces(gray_image):
    return face_detector(gray_image)


def get_face_landmarks(gray_image, face):
    return landmark_predictor(gray_image, face)


def detect_main_face(gray_image):
    faces = face_detector(gray_image)
    if not faces:
        return None
    return min(faces, key=lambda f: abs(f.center().x - gray_image.shape[1] // 2))