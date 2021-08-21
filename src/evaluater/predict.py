import argparse
import glob
import importlib
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd


def earth_movers_distance(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


class TrainDataGenerator(tf.keras.utils.Sequence):
    """inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator"""

    def __init__(self, samples, img_dir, batch_size, n_classes, basenet_preprocess, img_format,
                 img_load_dims=(256, 256), img_crop_dims=(224, 224), shuffle=True):
        self.samples = samples
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.basenet_preprocess = basenet_preprocess  # Keras basenet specific preprocessing function
        self.img_load_dims = img_load_dims  # dimensions that images get resized into when loaded
        self.img_crop_dims = img_crop_dims  # dimensions that images get randomly cropped to
        self.shuffle = shuffle
        self.img_format = img_format
        self.on_epoch_end()  # call ensures that samples are shuffled in first epoch if shuffle is set to True

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))  # number of batches per epoch

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        X, y = self.__data_generator(batch_samples)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generator(self, batch_samples):
        # initialize images and labels tensors for faster processing
        X = np.empty((len(batch_samples), *self.img_crop_dims, 3))
        y = np.empty((len(batch_samples), self.n_classes))

        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            img_file = os.path.join(self.img_dir, '{}.{}'.format(sample['image_id'], self.img_format))
            img = utils.load_image(img_file, self.img_load_dims)
            if img is not None:
                img = utils.random_crop(img, self.img_crop_dims)
                img = utils.random_horizontal_flip(img)
                X[i,] = img

            # normalize labels
            y[i,] = utils.normalize_labels(sample['label'])

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.basenet_preprocess(X)

        return X, y


class TestDataGenerator(tf.keras.utils.Sequence):
    '''inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator'''

    def __init__(self, samples, img_dir, batch_size, n_classes, basenet_preprocess, img_format,
                 img_load_dims=(224, 224)):
        self.samples = samples
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.basenet_preprocess = basenet_preprocess  # Keras basenet specific preprocessing function
        self.img_load_dims = img_load_dims  # dimensions that images get resized into when loaded
        self.img_format = img_format
        self.on_epoch_end()  # call ensures that samples are shuffled in first epoch if shuffle is set to True

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))  # number of batches per epoch

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        X, y = self.__data_generator(batch_samples)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))

    def __data_generator(self, batch_samples):
        # initialize images and labels tensors for faster processing
        X = np.empty((len(batch_samples), *self.img_load_dims, 3))
        y = np.empty((len(batch_samples), self.n_classes))

        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            img_file = os.path.join(self.img_dir, '{}.{}'.format(sample['image_id'], self.img_format))
            img = load_image(img_file, self.img_load_dims)
            if img is not None:
                X[i,] = img

            # normalize labels
            if sample.get('label') is not None:
                y[i,] = normalize_labels(sample['label'])

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.basenet_preprocess(X)

        return X, y


class Nima:
    def __init__(self, base_model_name, n_classes=10, learning_rate=0.001, dropout_rate=0, loss=earth_movers_distance,
                 decay=0, weights='imagenet'):
        self.n_classes = n_classes
        self.base_model_name = base_model_name
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.decay = decay
        self.weights = weights
        self._get_base_module()

    def _get_base_module(self):
        # import Keras base model module
        if self.base_model_name == 'InceptionV3':
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_v3')
        elif self.base_model_name == 'InceptionResNetV2':
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_resnet_v2')
        else:
            self.base_module = importlib.import_module('tensorflow.keras.applications.' + self.base_model_name.lower())

    def build(self):
        # get base model class
        BaseCnn = getattr(self.base_module, self.base_model_name)

        # load pre-trained model
        self.base_model = BaseCnn(input_shape=(224, 224, 3), weights=self.weights, include_top=False, pooling='avg')

        # add dropout and dense layer
        x = Dropout(self.dropout_rate)(self.base_model.output)
        x = Dense(units=self.n_classes, activation='softmax')(x)

        self.nima_model = Model(self.base_model.inputs, x)

    def compile(self):
        self.nima_model.compile(optimizer=Adam(lr=self.learning_rate, decay=self.decay), loss=self.loss)

    def preprocessing_function(self):
        return self.base_module.preprocess_input


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, target_file):
    with open(target_file, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def random_crop(img, crop_dims):
    h, w = img.shape[0], img.shape[1]
    ch, cw = crop_dims[0], crop_dims[1]
    assert h >= ch, 'image height is less than crop height'
    assert w >= cw, 'image width is less than crop width'
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    return img[y:(y + ch), x:(x + cw), :]


def random_horizontal_flip(img):
    assert len(img.shape) == 3, 'input tensor must have 3 dimensions (height, width, channels)'
    assert img.shape[2] == 3, 'image not in channels last format'
    if np.random.random() < 0.5:
        img = img.swapaxes(1, 0)
        img = img[::-1, ...]
        img = img.swapaxes(0, 1)
    return img


def load_image(img_file, target_size):
    return np.asarray(tf.keras.preprocessing.image.load_img(img_file, target_size=target_size))


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.' + img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})

    return samples


def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)


def main(base_model_name, weights_file, image_source, predictions_file, img_format='jpg'):
    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type='jpg')

    # build model and load weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(),
                                       img_format=img_format)

    # get predictions
    predictions = predict(nima.nima_model, data_generator)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction'] = calc_mean_score(predictions[i])

    print(json.dumps(samples, indent=2))

    # converts to CSV
    pd.read_json(json.dumps(samples)).to_csv(f'scores_{weights_file.split("_")[-2]}.csv', index=False)

    if predictions_file is not None:
        save_json(samples, predictions_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)
    parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default=None)

    args = parser.parse_args()

    main(**args.__dict__)
