import PIL
import os
import numpy as np
import random
from tensorflow.keras.utils import to_categorical


from specs import WIDTH, HEIGHT, CHANNEL


seed = 7
np.random.seed(seed)


def one_hot_encode(label) :
    return to_categorical(np.int32(list(label)), 10)


def load_data(path, train_ratio):
    datas = []
    labels = []

    with open(os.path.join(path, 'labels.txt'), 'r') as f:
        for i, line in enumerate(f):
            image_path = os.path.join(path, str(i) + ".png")
            chal_img = PIL.Image.open(image_path).resize((WIDTH, HEIGHT))
            data = np.array(chal_img).astype(np.float32)
            data = np.multiply(data, 1. / 255.)
            data = np.asarray(data)
            datas.append(data)
            labels.append(one_hot_encode(line.strip()))

    datas_labels = list(zip(datas, labels))
    random.shuffle(datas_labels)
    datas, labels = list(zip(*datas_labels))

    size = len(labels)
    train_size = int(size * train_ratio)

    train_datas = np.stack(datas[0:train_size])
    test_datas = np.stack(datas[train_size:size])
    train_labels = np.stack(labels[0:train_size])
    test_labels = np.stack(labels[train_size:size])

    return train_datas, train_labels, test_datas, test_labels

