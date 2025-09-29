import json
from pathlib import Path

filtered_dir = Path("data/scratch/filtered/dev")

segments = []
for ref in filtered_dir.glob("dev*_ref*"):
    noisy = str(ref).replace("_ref", "")

    fname = ref.stem.replace("_ref", "")
    ref = str(ref)

    print(fname.split("."))
    session, device, pid, _, start_end = fname.split(".")

    start, end = [int(a) for a in start_end.split("_")]
    duration = (end - start) / 16000

    segments.append(
        {
            "noisy": noisy,
            "ref": ref,
            "duration": duration,
            "session": session,
            "device": device,
            "target": pid,
            "start": start,
            "end": end,
        }
    )

with open("data/scratch/metadata/filtered.json", "w") as file:
    json.dump(segments, file, indent=4)
