import numpy as np
import torch
from torch.optim import Adam

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

from explore_data import makeDataSet, iterate_minibatches
from model import Network

path = 'data/trainSet/'
train_loader = iterate_minibatches
net = Network().cuda()

images, labels = makeDataSet(path)
# Original Ddata is 100 cropped to 90
labels = labels[:90]


def train(images, labels, epochs):
    optimizer = Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.SmoothL1Loss()
    epochs = 100
    print_every = 100
    error_l = []
    epochs_l = []
    for e in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader(images, labels, batchsize=12)):
            img_batch, label_batch = data
            img_batch = img_batch / 255.
            img_batch = torch.FloatTensor(img_batch)
            img_batch = img_batch.reshape(img_batch.shape[0], 3, 224, 224)
            label_batch = (label_batch - 100.)/50.
            label_batch = torch.FloatTensor(label_batch)
            label_batch = label_batch.view(label_batch.size(0), -1)
            output = net(img_batch.cuda())
            loss = criterion(output, label_batch.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % print_every == 0:
                print("Epochs-- {}/{} Loss-- {}".format(e+1,
                      epochs,
                      running_loss/print_every))
                error_l.append(running_loss/print_every)
                epochs_l.append(e+1)
    return output, error_l, epochs_l


pred, losses, epochs_l = train(images, labels, 50)


def visualize_pred(idx):
    x = images * 255
    x = images.reshape(images.shape[0], 224, 224, 3)
    x = x.cpu()
    x = x.detach().numpy()
    y = labels * 50. + 100.
    y = y.reshape(y.shape[0], 6, 2)
    y = y.cpu()
    y = y.detach().numpy()
    plt.imshow(x[idx].astype('uint8'), origin='lower')
    plt.scatter(y[idx][:, 0], y[idx][:, 1])


def predict(path):
    img = mpimg.imread(path)
    img = cv2.flip(img, 0)
    img = img/255.
    img = np.expand_dims(img, axis=0)
    img_tensor = torch.FloatTensor(img)
    img_tensor = img_tensor.reshape(img.shape[0], 3, 224, 224)
    output = net(img_tensor.cuda())
    output = output * 50. + 100.
    output = output.reshape(output.shape[0], 6, 2)
    output = output.cpu()
    output = output.detach().numpy()

    img_tensor = img_tensor.reshape(img.shape[0], 224, 224, 3)
    img_tensor = img_tensor.cpu()
    img_tensor = img_tensor.detach().numpy()
    img_tensor = img_tensor
    plt.imshow(np.squeeze(img_tensor), origin='lower')
    plt.scatter(output[0][:, 0], output[0][:, 1])


predict('data/testSet/train_x_0100.jpg')
