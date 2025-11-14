import hydra
import jiwer
import json
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    plot_wer(cfg)

    plt.savefig(f"transcripts/{cfg.listener}.png")


def plot_wer(cfg: DictConfig):

    base = "robbie"
    target = cfg.listener

    transformation = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
        ]
    )

    scores = {}

    for person in [base, target]:
        with open(f"transcripts/{person}.json", "r") as file:
            segments = {
                x["key"]: x for x in json.load(file)["segments"] if not x["isTrain"]
            }

        keys = sorted(segments.keys())

        scores[person] = {"ct": [], "baseline": []}
        all_scores = []
        for k in keys:
            session = k.split(".")[0]

            gt = segments[k]["ground_truth"]
            res = segments[k]["response"]

            wer = jiwer.wer(gt, res, transformation, transformation)

            if session in ["dev_03", "dev_04"]:
                scores[person]["ct"].append(wer)
            else:
                scores[person]["baseline"].append(wer)
            all_scores.append(wer)

    base_mean = np.mean(scores[base]["ct"] + scores[base]["baseline"])
    target_mean = np.mean(scores[target]["ct"] + scores[target]["baseline"])

    highest = max(all_scores)

    fig, ax = plt.subplots()

    ax.plot([0, highest], [0, highest], alpha=0.4)
    ax.scatter(scores[base]["ct"], scores[target]["ct"], color="darkred", label="ct")
    ax.scatter(
        scores[base]["baseline"],
        scores[target]["baseline"],
        color="tomato",
        label="baseline",
    )
    ax.set_title(f"{base} vs {target}")
    ax.set_xlabel(f"{base} WER %")
    ax.set_ylabel(f"{target} WER %")

    ax.axvline(base_mean, color="r", label=f"{base}={base_mean:.2%}")
    ax.axhline(target_mean, color="b", label=f"{target}={target_mean:.2%}")
    ax.legend()

    return fig


if __name__ == "__main__":
    main()
