from pathlib import Path
from pife import Pife

root = Path('/raid/imdb-clean/data/imdb-clean-1024')

paths = []
for path in root.iterdir():
    if path.is_dir():
        paths = paths + list(path.glob('*'))

output_dir = Path('/raid/imdb-clean-pife')

pife = Pife('retinaface', 2)
pife.extract(paths, output_dir)
