# Mini Listening Test

This should hopefully be quite straightforward...

```[bash]
# Using for the first time
git clone git@github.com:CHiME9-ECHI/ECHI-ListeningTests.git
cd ECHI-ListeningTests

# Pull if you already have it
cd ECHI-ListeningTests
git pull
```

To use the `uv` environment:

```[bash]
uv sync
source .venv/bin/activate
```

Or with `conda`:

```[bash]
# For a new environment
conda env create -f requirements.yaml
# To update an existing environment
conda env update -f requirements.yaml
```

The data is now included within the GitHub, so there's no need to download anything extra. To run the test:

```[bash]
streamlit run src/listening_test.py
```

There's about 48 segments to evaluate, and it goes pretty quickly (around 15 mins).

When you're done, it should show a plot comparing your scores to the noisy
transcripts by me.

When your done, push the results back to the GitHub (you might need to pull and
merge depending on who does it first).
