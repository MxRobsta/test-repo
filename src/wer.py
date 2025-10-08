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

    all_exps = [
        name
        for seg in segments
        for name in seg["transcripts"]
        if name != "ground_truth"
    ]
    all_exps = list(set(all_exps))

    for seg in segments:

        transcripts = {
            name: x for name, x in seg["transcripts"].items() if name != "ground_truth"
        }
        gt = seg["transcripts"]["ground_truth"]

        if not all(name in transcripts.keys() for name in all_exps):
            continue

        for person, ts in transcripts.items():
            w = jiwer.wer(gt, ts)

            if person not in wers:
                wers[person] = [w]
            else:
                wers[person].append(w)

    for person, wer in wers.items():
        plt.hist(wer)
        plt.title(person + f"{sum(wer)/len(wer):.2f}")
        plt.xlabel("WER")
        plt.ylabel("Frequency")
        plt.savefig(wer_dir / f"{person}.png")
        plt.close()

    plt.plot(
        [0, max(max(wers["robbie_noisy"]), max(wers["robbie_basesummed"]))],
        [0, max(max(wers["robbie_noisy"]), max(wers["robbie_basesummed"]))],
    )
    plt.scatter(wers["robbie_noisy"], wers["robbie_basesummed"])
    plt.title("Noisy vs RobbieBasesummed")
    plt.xlabel("WER %")
    plt.ylabel("WER %")
    plt.savefig(wer_dir / "scatter.png")


if __name__ == "__main__":
    main()
