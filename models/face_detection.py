from configs.model_paths import MODEL_PATHS
import dlib


face_detector = dlib.get_frontal_face_detector()

landmark_predictor = dlib.shape_predictor(MODEL_PATHS["shape_predictor"])


def detect_faces(image):
    return face_detector(image)

def get_landmarks(image, face):
    return landmark_predictor(image, face)