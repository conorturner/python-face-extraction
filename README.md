# Python Image Face Extraction

This python library accompanies the paper ___, allowing researchers and practitioners to access standardised and reliable face alignment for the projects.
To allow for ultimate flexibility the functionality must be accessed via minimal python scripts, as opposed to a command line interface. We make this design choice based on extensive experience with many face analysis dataset.

The following example extracts faces from the paths in 3 folders using the retinaface backend with a concurrency of 2. Concurrency is defined by your systems GPU memory and compute capacity, which can be discovered via trial and error, a concurrency of 1 will be suitable for almost any modern GPU.

## Get started

Our code is not published on PyPi so please clone the repo to get started.

```python
from pathlib import Path
from pife import Pife


output_dir = Path('/raid/chalearn15-pife')
paths = (
        list(Path('/raid/chalearn15/train').glob('*.jpg')) +
        list(Path('/raid/chalearn15/test').glob('*.jpg')) +
        list(Path('/raid/chalearn15/valid').glob('*.jpg'))
)

pife = Pife('retinaface', 2)
pife.extract(paths, output_dir)
```

The codebase was developed on an RTX8000 with 48GB of GPU memory and 64GB of CPU memory, such that the following concurrency levels are optimal.

```python
worker_map = {
    'opencv': 10,
    'dlib': 8,
    'mediapipe': 2,
    'mtcnn': 4,
    'retinaface': 2,
    'ssd': 4,
}
```

## Backends

| Backend   | Description                                            | Use Case             |
|-----------|--------------------------------------------------------|----------------------|
| opencv    | CPU based Haar Cascade Face and Eye Detector           | Simple/Fast/Unreliable |
| dlib | CPU based HOG model with regression tree eye detector. | Simple/Reliable/Medium Speed |
|  mediapipe   | E2E GPU based deep learning model                      | Very Fast/Medium Reliability |
|      mtcnn   | Robust and accuracy GPU model                          | Highly Reliable/Medium Speed |
|   retinaface | Most accurate and second most robust model, GPU based. | Slow/Accurate/Reliable |
|  ssd    | Deep learning face detector with opencv eye detector   | Unremarkable         |




If you find our code useful please consider citing our paper.

```
TBC
```