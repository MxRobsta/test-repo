import json
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

with open("data/mini_test/filtered.json", "r") as file:
    segments = json.load(file)

base_dir = Path("data/mini_test/base_nopad")
base_dir.mkdir(exist_ok=True)

noisy_dir = Path("data/mini_test/noisy_nopad")
noisy_dir.mkdir(exist_ok=True)

for seg in tqdm(segments):

    fname = Path(seg["noisy"]).name

    noisy, fs = sf.read(seg["noisy"])
    noisy = noisy[seg["start_pad"] : -seg["end_pad"]]
    sf.write(noisy_dir / fname, noisy, fs)

    base, fs = sf.read(seg["basesummed"])
    base = base[seg["start_pad"] : -seg["end_pad"]]
    sf.write(base_dir / fname, base, fs)  #

    seg["noisy_nopad"] = str(noisy_dir / fname)
    seg["base_nopad"] = str(base_dir / fname)

with open("data/mini_test/filtered.json", "w") as file:
    json.dump(segments, file, indent=2)
