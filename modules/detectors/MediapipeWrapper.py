# Link - https://google.github.io/mediapipe/solutions/face_detection
import numpy as np


def build_model():
    import mediapipe as mp  # this is not a must dependency. do not import it in the global level.
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
    return face_detection


mp_detector = build_model()


def detect_face(img, mode='eyes'):
    img_width = img.shape[1]
    img_height = img.shape[0]

    try:
        results = mp_detector.process(np.ascontiguousarray(img))
    except:
        print('\n\n shape', img.shape, '\n\n')

    if results.detections:
        detection = results.detections[0]
        bounding_box = detection.location_data.relative_bounding_box
        landmarks = detection.location_data.relative_keypoints

        x = int(bounding_box.xmin * img_width)
        w = int(bounding_box.width * img_width)
        y = int(bounding_box.ymin * img_height)
        h = int(bounding_box.height * img_height)

        right_eye = (int(landmarks[0].x * img_width), int(landmarks[0].y * img_height))
        left_eye = (int(landmarks[1].x * img_width), int(landmarks[1].y * img_height))

        if mode == 'eyes':
            if x > 0 and y > 0:
                return [right_eye, left_eye]
        elif mode == 'box':
            return x, y, w, h
        else:
            raise Exception(f'Unknown mode {mode}')

    return None
