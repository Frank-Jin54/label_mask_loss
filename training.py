# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from opt.partial_loss import MaskedCrossEntropyLoss, AdaptiveMaskedCrossEntropyLoss
import os
import pandas as pd
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)

import argparse

parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--epoch', '-e', dest='epoch', default=50, help='epoch')
parser.add_argument('--dataset', '-d', dest='dataset', default="CIFAR10", help='dataset', required=False)
parser.add_argument('--opt_alg', '-a', dest='opt_alg', default="SGD", help='opt_alg', required=False)
parser.add_argument('--lossfunction', '-l', dest='lossfunction', default="MASKEDLABEL", help='lossfunction', required=False)

args = parser.parse_args()


########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

current_folder = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_folder, 'data')
if args.dataset == "CIFAR10":
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

elif args.dataset == "CIFAR100":
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
elif args.dataset == "EMNIST":
    trainset = torchvision.datasets.EMNIST(root=data_path, train=True,
                                             download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.EMNIST(root=data_path, train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
elif args.dataset == "FashionMNIST":
    trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.FashionMNIST(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
elif args.dataset == "Caltech101":
    trainset = torchvision.datasets.Caltech101(root=data_path, train=True,
                                                 download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.Caltech101(root=data_path, train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
elif args.dataset == "CelebA":
    trainset = torchvision.datasets.CelebA(root=data_path, train=True,
                                               download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CelebA(root=data_path, train=False,
                                              download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
elif args.dataset == "Country211":
    trainset = torchvision.datasets.Country211(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.Country211(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
elif args.dataset == "DTD":
    trainset = torchvision.datasets.DTD(root=data_path, train=True,
                                               download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.DTD(root=data_path, train=False,
                                              download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
image_size = trainset.data.shape[1]
dataclasses_num = len(trainset.classes)

if args.lossfunction == "MASKEDLABEL":
    criterion = MaskedCrossEntropyLoss(alpha=0.9, num_class=dataclasses_num)
elif args.lossfunction == 'CROSSENTROPY':
    criterion = nn.CrossEntropyLoss()
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).


from model_define.hugging_face_vit import ViTForImageClassification

net = ViTForImageClassification(num_labels=dataclasses_num, imagesize=image_size)
device = torch.device('cuda:0')
net = net.to(device if device else "cpu")
from model_define.defined_model import Net
net =Net(num_class=dataclasses_num)

if args.opt_alg == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
elif args.opt_alg == "ADAM":
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
elif args.opt_alg == "RADAM":
    optimizer = optim.RAdam(net.parameters(), lr=1e-4)
elif args.opt_alg == "RMSprop":
    optimizer = optim.RMSprop(net.parameters(), lr=1e-4)
else:
    raise Exception("Not accept optimizer of {}".args.opt_alg)

########################################################################
def run_test(model_path):
    correct = 0
    total = 0
    net.load_state_dict(torch.load(model_path))
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def save_model(net, model_path):
    torch.save(net.state_dict(), model_path)
# 4. Train the network
acc = []
for epoch in range(int(args.epoch)):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs, interpolate_pos_encoding=True)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    model_path = os.path.join(current_folder, 'model', '{}_{}_{}_net.pth'.format(args.dataset, args.opt_alg, args.lossfunction))
    save_model(net, model_path)
    acc.append([epoch, run_test(model_path), round(running_loss, 2)])
print('Finished Training')
result_file = os.path.join(os.path.join(current_folder, 'result', 'result_{}_{}_{}.csv'.format(args.dataset, args.opt_alg, args.lossfunction)))
pd.DataFrame(acc).to_csv(result_file, header=["epoch", "training_acc", "training_loss"], index=False)