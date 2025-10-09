import json
import numpy as np

RNG = np.random.default_rng(123456789)

with open("src/filtered.json", "r") as file:
    segments = json.load(file)

choices = ["noisy_nopad", "base_nopad"]
counts = [0, 0]

for seg in segments:
    index = RNG.integers(0, 2)
    counts[index] += 1
    seg["random"] = seg[choices[index]]

with open("src/filtered.json", "w") as file:
    json.dump(segments, file, indent=2)

print(counts)
