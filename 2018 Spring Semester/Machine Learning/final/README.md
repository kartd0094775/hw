## Whale Classifier

### Requirements

```bash
numpy==1.14.5
pandas==0.21.0
keras==2.0.8
Pillow==5.1.0
scipy==1.0.0
scikit-image==0.13.1
tensorflow==1.1.0
unzip
wget
```

### Info

If directory "image" does not exist, the image dataset will be downloaded automatically before training or testing.

### Training Model

#### Basic usage

```bash
bash train.sh <output_path>
```

For example:

```bash
bash train.sh predict.csv
```

The output of training is the testing result by the single model trained this time.

### Predict Whale

#### Basic usage

```bash
bash test.sh <output_path>
```

For example:

```bash
bash test.sh predict.csv
```

## Bounding Box

### Requirements

```bash
numpy==1.14.5
keras==2.0.8
Pillow==5.1.0
scipy==1.0.0
scikit-image==0.13.1
tqdm==4.19.6
```

### Training Model

#### Basic usage

```bash
python3 bounding_box_train.py <label_data_path> <training_image_directory> <model_save_directory>
```

For example:

```bash
python3 bounding_box_train.py path/to/cropping.txt path/to/train path/to/model
```

#### Arguments

```bash
usage: bounding_box_train.py [-h] [--img_size IMG_SIZE]
                             [--max_rotation_angle MAX_ROTATION_ANGLE]
                             [--rotation_step ROTATION_STEP]
                             [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                             [--val_frac VAL_FRAC]
                             label_data_path training_img_dir model_save_dir
```

| Arguments            | Description                                                                                 | Type    | Default      |
| -------------------- | ------------------------------------------------------------------------------------------- | ------- | ------------ |
| label_data_path      | `cropping.txt` file path                                                                    | `str`   | required     |
| training_img_dir     | path to the directory where labeled images from Kaggle are placed                           | `str`   | required     |
| model_save_dir       | path to directory where checkpoint models are placed                                        | `str`   | required     |
| --img_size           | size of the images for training                                                             | `tuple` | `(128, 256)` |
| --max_rotation_angle | maximum rotation angle for data augmentation, set to `0` to train without data augmentation | `int`   | `10`         |
| --rotation_step      | step between rotation angle                                                                 | `int`   | `5`          |
| --batch_size         | training batch size                                                                         | `int`   | `32`         |
| --epochs             | training epochs                                                                             | `int`   | `500`        |
| --val_frac           | fraction of training data for validation                                                    | `float` | `0.1`        |

### Predict Bounding Box and Export Images

#### Basic usage

```bash
python3 bounding_box_predict.py <model_path> <image_dir> <export_directory>
```

For example:

```bash
python3 bounding_box_predict.py path/to/model.h5 path/to/images/dir path/to/export/images
```

#### Arguments

```bash
usage: bounding_box_predict.py [-h] [--model_img_size MODEL_IMG_SIZE]
                               [--orig_img_size ORIG_IMG_SIZE]
                               [--export_img_size EXPORT_IMG_SIZE]
                               model_path img_dir export_dir
```

| Arguments         | Description                                                                       | Type    | Default      |
| ----------------- | --------------------------------------------------------------------------------- | ------- | ------------ |
| model_path        | path to the model                                                                 | `str`   | required     |
| img_dir           | directory where images for applying bounding box are placed                       | `str`   | required     |
| export_dir        | directory where images are exported                                               | `str`   | required     |
| --model_img_size  | size of the images to model. It must be same as the input of model while training | `tuple` | `(128, 256)` |
| --orig_img_size   | size of the images before cropping                                                | `tuple` | `(320, 640)` |
| --export_img_size | size of the images to export                                                      | `tuple` | `(160, 320)` |
