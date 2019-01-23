if [ ! -d image ]; then
  # if directory "image" does not exist, download the image dataset.
  wget -O download.zip "https://www.dropbox.com/s/bz1bf08o0x33bhw/image.zip?dl=1"
  unzip download.zip
fi
python3 src/test.py
python3 src/voting_test.py
python3 src/find_equal.py data/train.csv predict/final_predict.csv $1
