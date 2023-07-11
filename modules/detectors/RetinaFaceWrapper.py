from retinaface import RetinaFace


def build_model():
    face_detector = RetinaFace.build_model()
    return face_detector


rf_detector = build_model()


def detect_face(img, mode='eyes'):
    # The BGR2RGB conversion will be done in the preprocessing step of retinaface.
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #retinaface expects RGB but OpenCV read BGR
    obj = RetinaFace.detect_faces(img, model=rf_detector, threshold=0.9)

    if type(obj) == dict:
        if 'face_1' not in obj:
            return

        face = obj['face_1']
        if mode == 'eyes':
            landmarks = face["landmarks"]
            return [landmarks["right_eye"], landmarks["left_eye"]]
        elif mode == 'box':
            left, top, right, bottom = tuple(face['facial_area'])
            return left, top, right - left, bottom - top
        else:
            raise Exception(f'Unknown mode {mode}')

    return None
