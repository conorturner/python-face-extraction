import cv2
from facenet_pytorch import MTCNN


# MTCNN requires CUDNN 8.6 which isn't meant to be compatiable with torch (it is really)
# use this to fix errors: pip install nvidia-cudnn-cu11==8.6.0.163


def build_model():
    face_detector = MTCNN(device='cuda')
    return face_detector


mtcnn_detector = build_model()


def detect_face(img, mode='eyes'):
    boxes, probs, landmarks = mtcnn_detector.detect(img, landmarks=True)

    if landmarks is None:
        return None

    if len(landmarks) > 0:
        if mode == 'eyes':
            detection = landmarks[0]
            left_eye = detection[0]
            right_eye = detection[1]

            return [left_eye, right_eye]
        elif mode == 'box':
            left, top, right, bottom = tuple(boxes[0])
            return left, top, right - left, bottom - top
        else:
            raise Exception(f'Unknown mode {mode}')

    return None
