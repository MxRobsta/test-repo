import csv
import hydra
import json
from omegaconf import DictConfig
from pathlib import Path

SAMPLE_RATE = 16000


def str_time_to_seconds(time_str):
    h, m, s = time_str.split(":")
    h = int(h)
    m = int(m)
    s = float(s)

    return s + m * 60 + h * 60 * 60


def dict_to_tsv(data, tsv_path):

    tsv_segs = []

    for seg in data:
        start = seg["start_time"]
        end = seg["end_time"]
        text = seg["text"]

        tsv_segs.append("\t".join([str(start), str(end), text]) + "\n")

    with open(tsv_path, "w") as file:
        file.writelines(tsv_segs)


@hydra.main(version_base=None, config_path="../config", config_name="process_rawts")
def main(cfg: DictConfig):

    # Get the target speakers
    sessions_fpath = cfg.sessions_file.format(dataset=cfg.dataset)
    with open(sessions_fpath, "r") as file:
        session_info = list(csv.DictReader(file))

    for sinfo in session_info:
        session = sinfo["session"]

        for i in range(1, 5):
            raw_fpath = cfg.raw_file.format(
                dataset=cfg.dataset, session=session, pos=f"pos{i}"
            )
            with open(raw_fpath, "r") as file:
                raw_ts = json.load(file)

            pid = sinfo[f"pos{i}"]

            tidy_ts = []

            for index, seg in enumerate(raw_ts):
                start_str = seg["start_time"]["original"]
                end_str = seg["end_time"]["original"]

                start_time = str_time_to_seconds(start_str)
                end_time = str_time_to_seconds(end_str)

                tidy_ts.append(
                    {
                        "index": index,
                        "pid": pid,
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_sample": int(SAMPLE_RATE * start_time),
                        "end_sample": int(SAMPLE_RATE * end_time),
                        "text": seg["words"],
                    }
                )

            tidy_fpath = Path(
                cfg.tidy_file.format(
                    dataset=cfg.dataset, session=session, pid=f"pos{i}"
                )
            )
            if not tidy_fpath.parent.exists():
                tidy_fpath.parent.mkdir(parents=True)
            with open(tidy_fpath, "w") as file:
                json.dump(tidy_ts, file, indent=4)
            tidy_fpath_pid = Path(
                cfg.tidy_file.format(dataset=cfg.dataset, session=session, pid=pid)
            )
            tidy_fpath_pid.unlink(missing_ok=True)
            tidy_fpath_pid.symlink_to(tidy_fpath)

            tsv_path = Path(
                cfg.tidy_tsv_file.format(
                    dataset=cfg.dataset, session=session, pid=f"pos{i}"
                )
            )
            tsv_path.parent.mkdir(parents=True, exist_ok=True)
            dict_to_tsv(tidy_ts, tsv_path)

            tsv_pid_path = Path(
                cfg.tidy_tsv_file.format(dataset=cfg.dataset, session=session, pid=pid)
            )
            tsv_pid_path.unlink(missing_ok=True)
            tsv_pid_path.symlink_to(tsv_path)


if __name__ == "__main__":
    main()
