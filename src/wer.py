import hydra
import jiwer
import json
import matplotlib.pyplot as plt
import numpy as np
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

    noisy_mean = np.mean(wers["robbie_noisy"])
    exp_mean = np.mean(wers[f"{cfg.listener}_{cfg.experiment}"])

    plt.plot(
        [
            0,
            max(
                max(wers["robbie_noisy"]), max(wers[f"{cfg.listener}_{cfg.experiment}"])
            ),
        ],
        [
            0,
            max(
                max(wers["robbie_noisy"]), max(wers[f"{cfg.listener}_{cfg.experiment}"])
            ),
        ],
    )
    plt.scatter(wers["robbie_noisy"], wers[f"{cfg.listener}_{cfg.experiment}"])
    plt.title(f"Noisy vs {cfg.listener}_{cfg.experiment}")
    plt.xlabel("Noisy WER %")
    plt.ylabel(f"{cfg.listener}_{cfg.experiment} WER %")

    plt.axvline(noisy_mean, color="r", label=f"Noisy={noisy_mean:.2%}")
    plt.axhline(exp_mean, label=f"Exp={exp_mean:.2%}")
    plt.legend()

    plt.savefig(wer_dir / "scatter.png")
    plt.show()


if __name__ == "__main__":
    main()
