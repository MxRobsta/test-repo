import hydra
import json
from omegaconf import DictConfig
from pathlib import Path
from pysepm.qualityMeasures import fwSNRseg, composite
from pystoi import stoi
import soundfile as sf
from tqdm import tqdm


def score_filtered(filtered_store):

    with open(filtered_store, "r") as file:
        filtered_segments = json.load(file)

    for seg in tqdm(filtered_segments):
        noisy, nfs = sf.read(seg["noisy"])
        ref, rfs = sf.read(seg["ref"])

        assert nfs == 16000
        assert rfs == 16000

        if noisy.shape[1] == 4:
            noisy = noisy[:, 0] + noisy[:, 1]
        else:
            noisy = noisy[:, 2]

        if "stoi" not in seg:
            seg["stoi"] = stoi(ref, noisy, nfs)
        if "fwsegsnr" not in seg:
            seg["fwsegsnr"] = fwSNRseg(ref, noisy, nfs)
        if "Csig" not in seg:
            csig, cbak, covl = composite(ref, noisy, nfs)
            seg["Csig"] = csig
            seg["Cbak"] = cbak
            seg["Covl"] = covl

    with open(filtered_store, "w") as file:
        json.dump(filtered_segments, file, indent=4)

    print_mets = ["stoi", "fwsegsnr", "Csig", "Cbak", "Covl"]

    for thing in filtered_segments:
        name = Path(thing["noisy"]).stem
        fstring = "{:<50}" + "{:<10.2f}" * len(print_mets)
        print(fstring.format(name, *[thing[met] for met in print_mets]))


@hydra.main(version_base=None, config_path="../config", config_name="score")
def main(cfg: DictConfig):
    score_filtered(cfg.filtered_store)


if __name__ == "__main__":
    main()
