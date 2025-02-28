import dlib

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
