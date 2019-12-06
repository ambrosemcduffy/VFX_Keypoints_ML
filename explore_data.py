import numpy as np
import cv2
import matplotlib.image as mpimg
import os
import json


def getFileNames(path):
    names_ = []
    if os.path.exists(path):
        names = os.listdir(path)
        for name in names:
            if name.endswith('.jpg'):
                names_.append(name)
    return names_


def import_images(path):
    names_ = getFileNames(path)
    images = []
    for image in names_:
        img = mpimg.imread(path+image)
        images.append(cv2.flip(img, 0))
    return np.array(images)


def get_targetData():
    with open('data/data.json', 'r') as f:
        data = json.load(f)
        len_data = len(data)
    target = []
    for i in range(len_data):
        target.append(data['Tracker'+str(i+1)])
    target = np.array(target)
    return target


def makeDataSet(img_path):
    targets = get_targetData()
    t = []
    for i in range(targets.shape[1]):
        t.append(targets[:, i, :])
    images = import_images(img_path)
    return images, np.array(t)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
