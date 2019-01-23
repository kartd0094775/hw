import argparse
import logging
import os

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from skimage.transform import resize
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='ML final: Cropping Images using Bounding Box Model for Humpback Whale Identification Challenge')
    # Path arguments
    parser.add_argument('model_path')
    parser.add_argument('img_dir')
    parser.add_argument('export_dir')
    # Image arguments
    parser.add_argument('--model_img_size', default=(128, 256), type=tuple)
    parser.add_argument('--orig_img_size', default=(320, 640), type=tuple)
    parser.add_argument('--export_img_size', default=(160, 320), type=tuple)

    args = parser.parse_args()
    return args


def ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)


def load_images(path, target_size):
    logging.info('Load images from {}'.format(path))
    images = []
    filenames = []
    for filename in tqdm(os.listdir(path)):
        img_path = os.path.join(path, filename)
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img) / 255.
        images.append(x)
        filenames.append(filename)
    return np.array(images), np.array(filenames)


def export(pred, filenames, img_dir, orig_img_size, export_img_size, export_dir):
    logging.info('Export images to {}'.format(export_dir))
    ensure_dir(export_dir)
    for i in tqdm(range(len(filenames))):
        img_path = os.path.join(img_dir, filenames[i])
        img = image.load_img(img_path, target_size=orig_img_size)
        img = img.crop((pred[i][1]*orig_img_size[1], pred[i][0]*orig_img_size[0],
                        pred[i][3]*orig_img_size[1], pred[i][2]*orig_img_size[0]))
        img = img.resize((export_img_size[1], export_img_size[0]))
        img.save(os.path.join(export_dir, filenames[i]))


def main():
    images, filenames = load_images(args.img_dir, args.model_img_size)

    logging.info('Load model from {}'.format(args.model_path))
    model = load_model(args.model_path)
    pred = model.predict(images, verbose=1)

    export(pred, filenames, args.img_dir, args.orig_img_size,
           args.export_img_size, args.export_dir)


if __name__ == '__main__':
    # Set logging config
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main()
