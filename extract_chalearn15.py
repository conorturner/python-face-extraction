from pathlib import Path
from pife import Pife


output_dir, paths = Path('/raid/chalearn15-pife'), (
        list(Path('/raid/chalearn15/train').glob('*.jpg')) +
        list(Path('/raid/chalearn15/test').glob('*.jpg')) +
        list(Path('/raid/chalearn15/valid').glob('*.jpg'))
)

pife = Pife('retinaface', 2)
pife.extract(paths, output_dir)
