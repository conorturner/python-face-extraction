from pathlib import Path

from pife import Pife

worker_map = {
    'opencv': 10,
    'dlib': 8,
    'mediapipe': 2,
    'mtcnn': 4,
    'retinaface': 2,
    'ssd': 4,
}

output_dir, paths = Path('/raid/pife-data/chalearn'), list(
    filter(lambda p: 'face' not in p.name, Path('/raid/chalearn/train').glob('*.jpg'))) + \
                    list(filter(lambda p: 'face' not in p.name, Path('/raid/chalearn/valid').glob('*.jpg'))) + \
                    list(filter(lambda p: 'face' not in p.name, Path('/raid/chalearn/test').glob('*.jpg')))
# output_dir, paths = Path('/raid/utk-pife'), list(Path('/raid/utk').glob('*'))
# output_dir, paths = Path('/raid/cfd-pife'), list(Path('/raid/cfd').glob('*'))
# output_dir, paths = Path('/raid/scut-fbp-pife'), list(Path('/home/campus.ncl.ac.uk/b4025068/datasets/SCUT-FBP5500_v2/Images').glob('*'))
# output_dir, paths = Path('/raid/morph2-pife'), list(Path('/raid/Album2').glob('*'))

for backend, n_worker in worker_map.items():
    pife = Pife(backend, n_worker)
    pife.extract(paths, output_dir)
