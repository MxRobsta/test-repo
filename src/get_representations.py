import hydra
import json
from omegaconf import DictConfig
from pathlib import Path
import ppgs
import torch
import torchaudio
from tqdm import tqdm


def get_representations(
    representation: str, filtered_store: str, output_dirtemplate: str
):

    with open(filtered_store, "r") as file:
        filtered_segments = json.load(file)

    output_dir = Path(
        output_dirtemplate.format(dataset="dev", representation=representation)
    )
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for seg in tqdm(filtered_segments):

        noisy = Path(seg["noisy"])
        ref = Path(seg["ref"])

        noisy_out = (output_dir / noisy.name).with_suffix(".pt")
        ref_out = (output_dir / ref.name).with_suffix(".pt")

        if representation == "ppg":
            for in_fpath, out_fpath in zip([noisy, ref], [noisy_out, ref_out]):
                audio, fs = torchaudio.load(in_fpath)
                assert fs == 16000
                if audio.shape[0] == 4:
                    audio = audio[:2, :].sum(dim=0, keepdim=True)
                elif audio.shape[0] == 7:
                    audio = audio.narrow(0, 2, 1)
                this_ppg = ppgs.from_audio(audio, fs)
                torch.save(this_ppg, out_fpath)

        else:
            raise ValueError(
                f"Representation {representation} not recognised. Add code here"
            )

        seg[f"noisy_{representation}"] = str(noisy_out)
        seg[f"ref_{representation}"] = str(ref_out)

    with open(filtered_store, "w") as file:
        json.dump(filtered_segments, file, indent=4)


@hydra.main(version_base=None, config_path="../config", config_name="representation")
def main(cfg: DictConfig):
    get_representations(
        cfg.representation_type, cfg.filtered_store, cfg.representation_dir
    )


if __name__ == "__main__":
    main()
