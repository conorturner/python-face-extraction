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

datasets = [
    {
        'out': Path('/raid/pife-box/chalearn'),
        'paths': list(
            filter(lambda p: 'face' not in p.name, Path('/raid/chalearn/train').glob('*.jpg'))) + \
                 list(filter(lambda p: 'face' not in p.name, Path('/raid/chalearn/valid').glob('*.jpg'))) + \
                 list(filter(lambda p: 'face' not in p.name, Path('/raid/chalearn/test').glob('*.jpg')))
    },
    {
        'out': Path('/raid/pife-box/utk'),
        'paths': list(Path('/raid/utk').glob('*'))
    },
    {
        'out': Path('/raid/pife-box/cfd'),
        'paths': list(Path('/raid/cfd').glob('*'))
    },
    {
        'out': Path('/raid/pife-box/scutfbp'),
        'paths': list(Path('/home/campus.ncl.ac.uk/b4025068/datasets/SCUT-FBP5500_v2/Images').glob('*'))
    },
    {
        'out': Path('/raid/pife-box/morph2'),
        'paths': list(Path('/raid/Album2').glob('*'))
    }
]

for dataset in datasets:
    output_dir, paths = dataset['out'], dataset['paths']
    print('\n\n', output_dir, '\n\n')

    for backend, n_worker in worker_map.items():
        pife = Pife(backend, n_worker)
        pife.extract(paths, output_dir, mode='box')
