import functools
import importlib
import os
from pathlib import Path
from typing import List
from multiprocessing.pool import ThreadPool, Pool
import torchvision.transforms as transforms
from tqdm import tqdm

from modules.extract import eye_based_align, make_square, padded_square_crop, to_tensor, box_align

os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import numpy as np
from PIL import Image

to_pil = transforms.ToPILImage()

detectors = {
    'opencv': 'modules.detectors.OpenCvWrapper',
    'dlib': 'modules.detectors.DlibWrapper',
    'mediapipe': 'modules.detectors.MediapipeWrapper',
    'mtcnn': 'modules.detectors.MtcnnTorchWrapper',
    'retinaface': 'modules.detectors.RetinaFaceWrapper',
    'ssd': 'modules.detectors.SsdWrapper',
}


def _load_image(path: Path) -> np.ndarray:
    img = np.asarray(Image.open(path))
    if img.shape[-1] == 1 or len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    return img


def _extract_image(output_dir: Path, path: Path, detector: str, scale=0.3, v_pos=0.45, mode='eyes'):
    out_path = output_dir / detector / path.name
    if path.suffix == '.pg':
        return

    if out_path.exists():
        return

    d = importlib.import_module(detectors[detector])
    img = _load_image(path)
    try:
        detection = d.detect_face(img, mode=mode)
    except:
        print('\n\nerror: ', str(path))
        raise

    if detection is None:
        return

    if mode == 'eyes':
        extracted = eye_based_align(img, np.array(detection[1]), np.array(detection[0]),
                                    desired_eye_ratio=scale, mid_eye_pos_ratio=v_pos)
    elif mode == 'box':
        extracted = box_align(img, detection)

    else:
        raise Exception(f'Unknown mode {mode}')

    if extracted is None:
        return

    if 0 in extracted.shape:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if '.' not in str(out_path):
            return
        if extracted.shape[0] == 4:
            extracted = extracted[:3, :, :]
        to_pil(extracted).save(str(out_path))
    except:
        print('error', path, extracted.shape, img.shape)
        raise


class Pife:
    def __init__(self, backend='opencv', n_worker=8):
        self.backend = backend
        self.n_worker = n_worker

    def extract(self, paths: List[Path], output_dir: Path, scale=0.3, v_pos=0.45, mode='eyes'):
        fn = functools.partial(_extract_image, output_dir, detector=self.backend,
                               scale=scale, v_pos=v_pos, mode=mode)
        with Pool(self.n_worker) as pool:
            for _ in tqdm(pool.imap_unordered(fn, paths), total=len(paths), desc=self.backend):
                pass
