# Mini Listening Test

This should hopefully be quite straightforward...

```[bash]
git clone git@github.com:CHiME9-ECHI/ECHI-ListeningTests.git
cd ECHI-ListeningTests
```

To use the `uv` environment:

```[bash]
uv venv
source .venv/bin/activate
uv sync
```

Or with `conda`:

```[bash]
conda env create -f requirements.yaml
```

Then finally, to run the listening test. This script downloads a subset of the
~45 segments from google drive (40MB in total) and plays them once and let's you
type in the words that you hear. You need `SoX` installed (this is how the audio
is played back).

> **Note:** You can't start typing until the audio has finished.

```[bash]
source mini_test.sh <NAME> <TYPE>
```

The `<TYPE>` parameter has a few options:

* `noisy_nopad` to listen to noisy segments
* `base_nopad` for the summed baseline segments
* `random` for a random sampling

There's about 45 segments to evaluate, and it goes pretty quickly (around 15 mins).

When you're done, it should show a plot comparing your scores to the noisy
transcripts by me.

When your done, push the results back to the GitHub (you might need to pull and
merge depending on who does it first).
