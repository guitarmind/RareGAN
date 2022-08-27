import os
import numpy as np
import tarfile
import _pickle as cPickle
from .dataset import Dataset

from PIL import Image
import cv2


def _unpickle_cifar10(file):
    return cPickle.load(file, encoding='latin1')


def _construct_dataset(data_x, data_y, data_high_fraction):
    data_y[data_y >= 1] = 1
    data_y = 1 - data_y
    if data_high_fraction is not None:
        num_low = np.where(data_y == 0)[0].shape[0]
        num_high = int(
            num_low / (1. - data_high_fraction) * data_high_fraction)
        if num_high == 0:
            num_high = 1

        filter_ = data_y == 0
        ids = np.random.permutation(np.where(data_y == 1)[0])
        high_selected = ids[:num_high]
        filter_[high_selected] = 1
        data_x = data_x[filter_]
        data_y = data_y[filter_]
    print("num high={}, num low={}".format(
        np.where(data_y == 1)[0].shape[0],
        np.where(data_y == 0)[0].shape[0]))
    dataset = Dataset()
    dataset.load_from_data(data_x, data_y)
    return dataset


def load_data(dataset, data_high_fraction=None):
    if dataset == 'MNIST':
        f = open(os.path.join('data', 'MNIST', "train-images-idx3-ubyte"))
        loaded = np.fromfile(file=f, dtype=np.uint8)
        data_x = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)
        data_x = data_x / 255.

        f = open(os.path.join('data', 'MNIST', 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=f, dtype=np.uint8)
        data_y = loaded[8:].reshape((60000)).astype(np.int32)

        return _construct_dataset(data_x, data_y, data_high_fraction)
    elif dataset == 'CIFAR10':
        tar = tarfile.open(os.path.join(
            'data', 'CIFAR10', "cifar-10-python.tar.gz"))
        data_x = []
        data_y = []
        for i in range(1, 6):
            file = tar.extractfile(
                os.path.join("cifar-10-batches-py", "data_batch_{}".format(i)))
            dict_ = _unpickle_cifar10(file)
            sub_data_x = dict_["data"]
            sub_data_y = np.asarray(dict_["labels"], dtype=np.int32)
            assert list(sub_data_x.shape) == [10000, 3072]
            assert sub_data_x.dtype == np.uint8
            assert list(sub_data_y.shape) == [10000]
            data_x.append(sub_data_x)
            data_y.append(sub_data_y)

        data_x = np.concatenate(data_x, axis=0)
        assert list(data_x.shape) == [50000, 3072]
        data_y = np.concatenate(data_y, axis=0)
        assert list(data_y.shape) == [50000]

        data_x = np.reshape(data_x, [50000, 3, 32, 32])
        data_x = np.transpose(data_x, [0, 2, 3, 1])
        data_x = data_x.astype(np.float64)
        data_x = data_x / 255.  # -1~1
        return _construct_dataset(data_x, data_y, data_high_fraction)

    elif dataset == 'KolektorSDD':
        dataset_folder = "/workspace/Kaggle/KolektorSDD"

        # https://stackoverflow.com/questions/4808221/is-there-a-bounding-box-function-slice-with-non-zero-values-for-a-ndarray-in
        def crop_bbox(img):
            rows = np.any(img, axis=1)
            cols = np.any(img, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return img[ymin:ymax + 1, xmin:xmax + 1], (ymin, ymax, xmin, xmax)

        dataset_inputs = {}
        dataset_labels = {}
        defective_count = 0
        for folder in os.listdir(dataset_folder):
            for filename in os.listdir(f"{dataset_folder}/{folder}"):

                if ".bmp" in filename:
                    part_id = filename.split("_")[0]

                    label_img = Image.open(f"{dataset_folder}/{folder}/{filename}")
                    label_img = np.array(label_img)

                    if np.sum(label_img) > 0:
                        defective_count += 1

                        # print(f"{folder}-{part_id} has defect:",
                        # np.sum(label_img) > 0)

                        dataset_labels[f"{folder}-{part_id}"] = label_img

                else:
                    part_id = filename.split(".")[0]

                    input_img = Image.open(
                        f"{dataset_folder}/{folder}/{filename}").convert('L')
                    input_img = np.array(input_img)

                    dataset_inputs[f"{folder}-{part_id}"] = input_img

        data_x = []
        data_y = []
        for key, input_image in dataset_inputs.items():

            if key in dataset_labels:
                label_image = dataset_labels[key]

                # Crop center 500px
                bbox_image, (ymin, ymax, xmin, xmax) = crop_bbox(label_image)

                y_center = round((ymax + ymin) / 2)
                x_center = round((xmax + xmin) / 2)

                input_image = input_image[y_center - 250:y_center + 250, :]

                input_image = cv2.resize(
                    input_image, (32, 32)).astype(np.float32)

                assert input_image.shape == (32, 32)

                data_x.append(input_image[np.newaxis, :, :, np.newaxis] / 255)
                data_y.append(1)
            else:
                y_center = round(input_image.shape[0] / 2)

                input_image = input_image[y_center - 250:y_center + 250, :]

                input_image = cv2.resize(
                    input_image, (32, 32)).astype(np.float32)

                assert input_image.shape == (32, 32)

                data_x.append(input_image[np.newaxis, :, :, np.newaxis] / 255)
                data_y.append(2)

        data_x = np.concatenate(data_x, axis=0)
        data_y = np.array(data_y, dtype=np.int32)

        return _construct_dataset(data_x, data_y, data_high_fraction)
