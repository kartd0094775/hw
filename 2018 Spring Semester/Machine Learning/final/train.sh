if [ ! -d image ]; then
  # if directory "image" does not exist, download the image dataset.
  wget -O download.zip "https://www.dropbox.com/s/bz1bf08o0x33bhw/image.zip?dl=1"
  unzip download.zip
fi
python3 src/train.py
python3 src/voting_train.py
python3 src/find_equal.py data/train.csv predict/final_predict_train.csv $1
