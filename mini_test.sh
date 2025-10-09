listener=$1
experiment=$2

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    conda activate echi-listeningtest
fi

if [ ! -d "data/mini_test" ]; then
    gdown --fuzzy "https://drive.google.com/file/d/1w1Sg9KJGbfVR81EG8lrZA7pYH0GF9QKQ/view?usp=sharing" -O data/mini_test.tar.gz
    tar -xvzf data/mini_test.tar.gz -C data/
fi

python3 src/mini_test.py listener=${listener} experiment=${experiment}
python3 src/wer.py listener=${listener} experiment=${experiment}