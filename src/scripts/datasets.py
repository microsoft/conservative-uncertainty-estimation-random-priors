"""
Helper functions to load the different data sets in the uncertainties project.

Copyright 2019
Vincent Fortuin
Microsoft Research Cambridge
"""

import numpy as np
import pickle
from core import *
from torch_backend import *


def get_cifar10_data(data_dir="./data", exclude_classes=None):
    dataset = cifar10(root=data_dir)
    data_mean = np.mean(dataset['train']['data'], axis=(0,1,2))/255
    data_std = np.std(dataset['train']['data'], axis=(0,1,2))/255
    train_set = list(zip(transpose(normalise(dataset['train']['data'],
                            mean=data_mean, std=data_std)), dataset['train']['labels']))
    test_set = list(zip(transpose(normalise(dataset['test']['data'],
                            mean=data_mean, std=data_std)), dataset['test']['labels']))
    excluded_set = []
    if exclude_classes is not None:
        excluded_set.extend([(x,y) for x,y in train_set if y in exclude_classes])
        excluded_set.extend([(x,y) for x,y in test_set if y in exclude_classes])
        train_set = [(x,y) for x,y in train_set if y not in exclude_classes]
        test_set = [(x,y) for x,y in test_set if y not in exclude_classes]
    return train_set, test_set, excluded_set


def get_cifar100_data(data_dir="./data", exclude_classes=None):
    data_train = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
    data_test = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
    dataset = {"train": {"data": data_train.data, "labels": data_train.targets},
               "test": {"data": data_test.data, "labels": data_test.targets}}
    data_mean = np.mean(dataset['train']['data'], axis=(0,1,2))/255
    data_std = np.std(dataset['train']['data'], axis=(0,1,2))/255
    train_set = list(zip(transpose(normalise(dataset['train']['data'],
                                mean=data_mean, std=data_std)), dataset['train']['labels']))
    test_set = list(zip(transpose(normalise(dataset['test']['data'],
                                mean=data_mean, std=data_std)), dataset['test']['labels']))
    excluded_set = []
    if exclude_classes is not None:
        excluded_set.extend([(x,y) for x,y in train_set if y in exclude_classes])
        excluded_set.extend([(x,y) for x,y in test_set if y in exclude_classes])
        train_set = [(x,y) for x,y in train_set if y not in exclude_classes]
        test_set = [(x,y) for x,y in test_set if y not in exclude_classes]
    return train_set, test_set, excluded_set


def get_svhn_data(data_dir="./data", exclude_classes=None):
    data_train = torchvision.datasets.SVHN(root=data_dir, split="train", download=True)
    data_test = torchvision.datasets.SVHN(root=data_dir, split="test", download=True)
    dataset = {"train": {"data": data_train.data.transpose(0,2,3,1), "labels": data_train.labels},
               "test": {"data": data_test.data.transpose(0,2,3,1), "labels": data_test.labels}}
    data_mean = np.mean(dataset['train']['data'], axis=(0,1,2))/255
    data_std = np.std(dataset['train']['data'], axis=(0,1,2))/255
    train_set = list(zip(transpose(normalise(dataset['train']['data'],
                                mean=data_mean, std=data_std)), dataset['train']['labels']))
    test_set = list(zip(transpose(normalise(dataset['test']['data'],
                                mean=data_mean, std=data_std)), dataset['test']['labels']))
    excluded_set = []
    if exclude_classes is not None:
        excluded_set.extend([(x,y) for x,y in train_set if y in exclude_classes])
        excluded_set.extend([(x,y) for x,y in test_set if y in exclude_classes])
        train_set = [(x,y) for x,y in train_set if y not in exclude_classes]
        test_set = [(x,y) for x,y in test_set if y not in exclude_classes]
    return train_set, test_set, excluded_set


def get_imagenet_data(data_dir="./data", exclude_classes=None):
    data_train = []
    for i in range(1,11):
        with open(f"{data_dir}/imagenet32/train_data_batch_{i}", "rb") as infile:
            data_train.append(pickle.load(infile))
    data_train = {"data": np.concatenate([batch['data'] for batch in data_train]), "labels": np.concatenate([batch['labels'] for batch in data_train])}
    with open(f"{data_dir}/imagenet32/val_data", "rb") as infile:
        data_test = pickle.load(infile)
    dataset = {"train": {"data": data_train['data'].reshape([-1, 3, 32, 32]).transpose([0,2,3,1]), "labels": np.array(data_train['labels'])},
               "test": {"data": data_test['data'].reshape([-1, 3, 32, 32]).transpose([0,2,3,1]), "labels": np.array(data_test['labels'])}}
    data_mean = np.mean(dataset['train']['data'], axis=(0,1,2))/255
    data_std = np.std(dataset['train']['data'], axis=(0,1,2))/255
    train_set = list(zip(transpose(normalise(dataset['train']['data'],
                                mean=data_mean, std=data_std)), dataset['train']['labels']))
    test_set = list(zip(transpose(normalise(dataset['test']['data'],
                                mean=data_mean, std=data_std)), dataset['test']['labels']))
    excluded_set = []
    if exclude_classes is not None:
        excluded_set.extend([(x,y) for x,y in train_set if y in exclude_classes])
        excluded_set.extend([(x,y) for x,y in test_set if y in exclude_classes])
        train_set = [(x,y) for x,y in train_set if y not in exclude_classes]
        test_set = [(x,y) for x,y in test_set if y not in exclude_classes]
    return train_set, test_set, excluded_set


def get_cifar10_onelabel_data(data_dir="./data", exclude_classes=None):
    dataset = cifar10(root=data_dir)
    data_mean = np.mean(dataset['train']['data'], axis=(0,1,2))/255
    data_std = np.std(dataset['train']['data'], axis=(0,1,2))/255
    train_set = list(zip(transpose(normalise(dataset['train']['data'],
                            mean=data_mean, std=data_std)), np.ones_like(dataset['train']['labels'])))
    test_set = list(zip(transpose(normalise(dataset['test']['data'],
                            mean=data_mean, std=data_std)), np.ones_like(dataset['test']['labels'])))
    excluded_set = []
    if exclude_classes is not None:
        excluded_set.extend([(x,y) for x,y in train_set if y in exclude_classes])
        excluded_set.extend([(x,y) for x,y in test_set if y in exclude_classes])
        train_set = [(x,y) for x,y in train_set if y not in exclude_classes]
        test_set = [(x,y) for x,y in test_set if y not in exclude_classes]
    return train_set, test_set, excluded_set


def get_cifar10_reduced_data(data_dir="./data", exclude_classes=None, reduced_size=75):
    dataset = cifar10(root=data_dir)
    data_mean = np.mean(dataset['train']['data'], axis=(0,1,2))/255
    data_std = np.std(dataset['train']['data'], axis=(0,1,2))/255
    train_set = list(zip(transpose(normalise(dataset['train']['data'],
                            mean=data_mean, std=data_std)), dataset['train']['labels']))
    test_set = list(zip(transpose(normalise(dataset['test']['data'],
                            mean=data_mean, std=data_std)), dataset['test']['labels']))
    excluded_set = []
    if exclude_classes is not None:
        excluded_set.extend([(x,y) for x,y in train_set if y in exclude_classes])
        excluded_set.extend([(x,y) for x,y in test_set if y in exclude_classes])
        train_set = [(x,y) for x,y in train_set if y not in exclude_classes]
        test_set = [(x,y) for x,y in test_set if y not in exclude_classes]
    subset_indices = np.arange(reduced_size)
    train_set = [elem for i, elem in enumerate(train_set) if i in subset_indices]
    return train_set, test_set, excluded_set


def get_imagenet_reduced_data(data_dir="./data", exclude_classes=None, reduced_size=75):
    data_train = []
    for i in range(1,11):
        with open(f"{data_dir}/imagenet32/train_data_batch_{i}", "rb") as infile:
            data_train.append(pickle.load(infile))
    data_train = {"data": np.concatenate([batch['data'] for batch in data_train]), "labels": np.concatenate([batch['labels'] for batch in data_train])}
    with open(f"{data_dir}/imagenet32/val_data", "rb") as infile:
        data_test = pickle.load(infile)
    dataset = {"train": {"data": data_train['data'].reshape([-1, 3, 32, 32]).transpose([0,2,3,1]), "labels": np.array(data_train['labels'])},
               "test": {"data": data_test['data'].reshape([-1, 3, 32, 32]).transpose([0,2,3,1]), "labels": np.array(data_test['labels'])}}
    data_mean = np.mean(dataset['train']['data'], axis=(0,1,2))/255
    data_std = np.std(dataset['train']['data'], axis=(0,1,2))/255
    train_set = list(zip(transpose(normalise(dataset['train']['data'],
                                mean=data_mean, std=data_std)), dataset['train']['labels']))
    test_set = list(zip(transpose(normalise(dataset['test']['data'],
                                mean=data_mean, std=data_std)), dataset['test']['labels']))
    excluded_set = []
    if exclude_classes is not None:
        excluded_set.extend([(x,y) for x,y in train_set if y in exclude_classes])
        excluded_set.extend([(x,y) for x,y in test_set if y in exclude_classes])
        train_set = [(x,y) for x,y in train_set if y not in exclude_classes]
        test_set = [(x,y) for x,y in test_set if y not in exclude_classes]
    subset_indices = np.arange(reduced_size)
    train_set = [elem for i, elem in enumerate(train_set) if i in subset_indices]
    return train_set, test_set, excluded_set