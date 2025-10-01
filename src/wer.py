import hydra
import jiwer
import json
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from pathlib import Path


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):

    filtered_store = cfg.filtered_store
    wer_dir = Path(cfg.paths.scratch_dir) / "metadata/WER"
    wer_dir.mkdir(exist_ok=True, parents=True)

    with open(filtered_store, "r") as file:
        segments = json.load(file)

    wers = {}

    for seg in segments:

        transcripts = {
            name: x for name, x in seg["transcripts"].items() if name != "ground_truth"
        }
        gt = seg["transcripts"]["ground_truth"]

        for person, ts in transcripts.items():
            w = jiwer.wer(gt, ts)

            if person not in wers:
                wers[person] = [w]
            else:
                wers[person].append(w)

    for person, wer in wers.items():
        plt.hist(wer)
        plt.title(person)
        plt.xlabel("WER")
        plt.ylabel("Frequency")
        plt.savefig(wer_dir / f"{person}.png")


if __name__ == "__main__":
    main()
