import cv2
from mtcnn import MTCNN


# MTCNN requires CUDNN 8.6 which isn't meant to be compatiable with torch (it is really)
# use this to fix errors: pip install nvidia-cudnn-cu11==8.6.0.163


def build_model():
    face_detector = MTCNN()
    return face_detector


mtcnn_detector = build_model()


def detect_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mtcnn expects RGB but OpenCV read BGR
    detections = mtcnn_detector.detect_faces(img_rgb)

    if len(detections) > 0:
        detection = detections[0]
        keypoints = detection["keypoints"]
        left_eye = keypoints["left_eye"]
        right_eye = keypoints["right_eye"]

        return [left_eye, right_eye]

    return None
