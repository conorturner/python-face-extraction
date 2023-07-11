from pathlib import Path
import gdown
import bz2
import os

import numpy as np


def get_deepface_home():
    return str(os.getenv("DEEPFACE_HOME", default=str(Path.home())))


def build_model():
    home = get_deepface_home()

    import dlib  # this requirement is not a must that's why imported here

    # check required file exists in the home/.deepface/weights folder
    if os.path.isfile(home + '/.deepface/weights/shape_predictor_5_face_landmarks.dat') != True:
        print("shape_predictor_5_face_landmarks.dat.bz2 is going to be downloaded")

        url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
        output = home + '/.deepface/weights/' + url.split("/")[-1]

        gdown.download(url, output, quiet=False)

        zipfile = bz2.BZ2File(output)
        data = zipfile.read()
        newfilepath = output[:-4]  # discard .bz2 extension
        open(newfilepath, 'wb').write(data)

    face_detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(home + "/.deepface/weights/shape_predictor_5_face_landmarks.dat")

    detector = {}
    detector["face_detector"] = face_detector
    detector["sp"] = sp
    return detector


dlib_detector = build_model()


def detect_face(img, mode='eyes'):
    sp = dlib_detector["sp"]
    face_detector = dlib_detector["face_detector"]

    try:
        detections = face_detector(img, 1)
    except:
        print(img.shape, img.dtype)
        raise

    if len(detections) > 0:
        d = detections[0]
        left = d.left()
        right = d.right()
        top = d.top()
        bottom = d.bottom()

        if mode == 'eyes':
            img_shape = sp(img, d)

            lm = np.array([(p.x, p.y) for p in img_shape.parts()])
            left_eye = (lm[0] + lm[1]) / 2
            right_eye = (lm[2] + lm[3]) / 2

            return [right_eye, left_eye]
        elif mode == 'box':
            return left, top, right - left, bottom - top
        else:
            raise Exception(f'Unknown mode {mode}')

    return None
