"""
Script to recreate the figures on CIFAR-10 from the trained models.

Copyright 2019
Vincent Fortuin
Microsoft Research Cambridge
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from glob import glob
from datetime import datetime
from sklearn.metrics import roc_auc_score
from core import *
from torch_backend import *

import models
import datasets

sns.set(context="paper", style="whitegrid", font_scale=1.5)

ex_name = f"cifar10"
date_string = datetime.today().strftime("%y%m%d")

Bs = [1, 5, 10]
batch_size = 64

print("Loading models...")

deep_ensemble_noadv_paths = glob("../../logs/*_DE_*")
deep_ensemble_adv_paths = glob("../../logs/*_DEAT_*")
uncertainty_ensemble_paths = glob("../../logs/*_RP_*")
dropout_paths = glob("../../logs/*_DR_*")

models_all = {"RP": {}, "DE": {}, "DE+AT": {}, "DR": {}}

np.random.seed(0)

for model_type in models_all.keys():
    if model_type == "RP":
        path = uncertainty_ensemble_paths
    elif model_type == "DE":
        path = deep_ensemble_noadv_paths
    elif model_type == "DE+AT":
        path = deep_ensemble_adv_paths
    elif model_type == "DR":
        path = dropout_paths

    for B in Bs:
        if model_type == "DR":
            model = torch.load(f"{path[0]}/models/uncertainty_model")
            model.bootstrap_size = B
        else:
            model_paths = np.random.choice(path, size=B, replace=False)
            models = [torch.load(f"{model_path}/models/uncertainty_model") for model_path in model_paths]
            model = torch.load(f"{path[0]}/models/uncertainty_model")
            model.ensemble_size = B
            model.ensemble = [single_model.ensemble[0] for single_model in models]
        models_all[model_type][str(B)] = model

print("Loading data...")

DATA_DIR = "./data"

train_set, test_set, excluded_set = datasets.get_cifar10_data(data_dir=DATA_DIR, exclude_classes=[0,1,3,4,8,9])
_, _, cats_deer_set = datasets.get_cifar10_data(data_dir=DATA_DIR, exclude_classes=[3,4])
_, _, vehicles_set = datasets.get_cifar10_data(data_dir=DATA_DIR, exclude_classes=[0,1,8,9])
_, svhn_set, _ = datasets.get_svhn_data(data_dir=DATA_DIR)

np.random.seed(0)

def get_reduced_set(data_set, sample_size):
    reduced_indices = np.random.choice(len(data_set), size=sample_size, replace=False)
    reduced_set = [elem for i, elem in enumerate(data_set) if i in reduced_indices]
    return reduced_set

sample_size = np.min([len(train_set), len(test_set), len(excluded_set), len(svhn_set)])

train_set = get_reduced_set(train_set, sample_size)
test_set = get_reduced_set(test_set, sample_size)
excluded_set = get_reduced_set(excluded_set, sample_size)
svhn_set = get_reduced_set(svhn_set, sample_size)

data_sets = {"train": train_set,
            "test": test_set,
             "excluded": excluded_set,
            "cats_deer": cats_deer_set,
            "vehicles": vehicles_set,
            "svhn": svhn_set}

batches = {k: Batches(v, batch_size, shuffle=False, drop_last=True) for k,v in data_sets.items()}

print("Making predictions...")

predictions_all = {}
for model_type in models_all.keys():
    predictions_model = {}
    for B in Bs:
        B = str(B)
        predictions_B = {}
        for data_set, batch_iter in batches.items():
            predictions_data = {}
            print(f"{model_type}, {B}, {data_set}")
            predictions = []
            uncertainties = []
            for batch in batch_iter:
                uncertainties.extend(models_all[model_type][B](batch)['uncertainties'].cpu().detach().numpy())
                if model_type == "RP":
                    predictions.extend(models_all["DE+AT"]["1"](batch)['correct'].cpu().detach().numpy())
                else:
                    predictions.extend(models_all[model_type][B](batch)['correct'].cpu().detach().numpy())
            predictions_data['predictions'] = np.array(predictions)
            predictions_data['uncertainties'] = np.array(uncertainties)
            predictions_B[data_set] = predictions_data
        predictions_model[B] = predictions_B

        del models_all[model_type][B]
        torch.cuda.empty_cache()

    predictions_all[model_type] = predictions_model

with open(f"../../logs/{date_string}_{ex_name}_comparison.pkl", "wb") as outfile:
    pickle.dump(predictions_all, outfile)

## Make plots

#### Uncertainties

with open(f"../../logs/{date_string}_{ex_name}_comparison.pkl", "rb") as infile:
    predictions_all = pickle.load(infile)

print("Plotting figures...")

fig, axes = plt.subplots(nrows=len(Bs), ncols=len(predictions_all), figsize=(5*len(predictions_all),5*len(Bs)))

sample_size = batch_size

for B, axs in zip(Bs, axes):
    for model_type, ax in zip(predictions_all.keys(), axs):
        ax.title.set_text(f"{model_type}, B = {B}")
        aucs = {}
        for data_set in predictions_all[model_type][str(B)].keys():
            uncs_cat = np.concatenate([predictions_all[model_type][str(B)]['train']['uncertainties'],
                                      predictions_all[model_type][str(B)][data_set]['uncertainties']])
            labels_cat = np.concatenate([np.zeros_like(predictions_all[model_type][str(B)]['train']['uncertainties']),
                                         np.ones_like(predictions_all[model_type][str(B)][data_set]['uncertainties'])])
            aucs[data_set] = roc_auc_score(labels_cat, uncs_cat)

        for i, data_set in enumerate(predictions_all[model_type][str(B)].keys()):
            unc = predictions_all[model_type][str(B)][data_set]['uncertainties']
            sns.scatterplot(np.arange(sample_size*i, sample_size*(i+1)), np.random.choice(unc, size=sample_size, replace=False),
                            edgecolor=None, s=1.0, label=f"{data_set} (AUC = {aucs[data_set]:.2f})", ax=ax)
            sns.lineplot([sample_size*i, sample_size*(i+1)], [np.mean(unc), np.mean(unc)], linewidth=3, ax=ax)
        ax.set_xlabel("data")
        ax.set_ylabel("estimated uncertainty")
        ax.legend(markerscale=5)

fig.tight_layout()

fig.savefig(f"../../figures/{date_string}_{ex_name}_uncertainties_all.pdf")
fig.savefig(f"../../figures/{date_string}_{ex_name}_uncertainties_all.png")

#### Distributions

def gaussian_kde(data, bandwidth=0.1, gridsize=1000, xlim=1.):
    kde = np.zeros(gridsize)
    xs = np.linspace(0,xlim,gridsize)
    gaussian = (1/(np.sqrt(2*np.pi)*bandwidth))*np.exp(-np.linspace(-xlim,xlim,gridsize*2)**2/(2*(bandwidth**2)))
    for value in data:
        val_bin = np.where(value <= xs)[0][0]
        kde += gaussian[gridsize-val_bin:2*gridsize-val_bin]
    kde /= np.sum(kde)
    return kde

fig, axes = plt.subplots(nrows=2, ncols=len(predictions_all), figsize=(15,5))

bandwidth_rp =  0.3
bandwidth_de = 0.02
gridsize = 1000
cut = 0

axs = axes[0]
for i, model_type in enumerate(predictions_all.keys()):
    for j, B in enumerate(Bs):
        uncertainties_seen = predictions_all[model_type][str(B)]["train"]['uncertainties']
        uncertainties_unseen = np.concatenate([predictions_all[model_type][str(B)]["test"]['uncertainties'],
                                               predictions_all[model_type][str(B)]["excluded"]['uncertainties']])

        xlim = np.max([np.max(uncertainties_seen), np.max(uncertainties_unseen), 1.])
        xs = np.linspace(0,xlim,gridsize)

        if model_type == "RP":
            bandwidth = bandwidth_rp
        else:
            bandwidth = bandwidth_de

        density = gaussian_kde(uncertainties_seen, bandwidth=bandwidth, gridsize=gridsize, xlim=xlim)
        sns.lineplot(xs,density, ax=axes[0][i], color=sns.color_palette("Blues", 4)[j+1], label=f"B = {B}")

        density = gaussian_kde(uncertainties_unseen, bandwidth=bandwidth, gridsize=gridsize, xlim=xlim)
        sns.lineplot(xs,density, ax=axes[1][i], color=sns.color_palette("Greens", 4)[j+1], label=f"B = {B}")


    ax.set_xlabel("uncertainty")
    ax.set_ylabel("frequency")
    ax.legend(markerscale=20)

    axes[0][i].set_xlim([-1e-4,xlim])
    axes[1][i].set_xlim([-1e-4,xlim])

    axes[0][i].title.set_text(f"{model_type}, seen")
    axes[1][i].title.set_text(f"{model_type}, unseen")

fig.tight_layout()

fig.savefig(f"../../figures/{date_string}_{ex_name}_distributions.pdf")
fig.savefig(f"../../figures/{date_string}_{ex_name}_distributions.png")

#### Accuracy curves

accuracies_sorted_all = {}

for model_type in predictions_all.keys():
    accuracies_model = {}
    for B in predictions_all[model_type].keys():
        uncertainties_all = np.concatenate([predictions_all[model_type][B]["train"]["uncertainties"],
                             predictions_all[model_type][B]["test"]["uncertainties"],
                             predictions_all[model_type][B]["excluded"]["uncertainties"]])
        preds_all = np.concatenate([predictions_all[model_type][B]["train"]["predictions"],
                             predictions_all[model_type][B]["test"]["predictions"],
                             predictions_all[model_type][B]["excluded"]["predictions"]])
        uncertainties_sort = np.argsort(uncertainties_all)
        uncertainties_sorted = uncertainties_all[uncertainties_sort]
        predictions_sorted = preds_all[uncertainties_sort]
        accuracy_sorted = np.cumsum(predictions_sorted) / np.arange(1, len(predictions_sorted)+1)
        accuracies_model[B] = accuracy_sorted
    accuracies_sorted_all[model_type] = accuracies_model

fig, axs = plt.subplots(nrows=1, ncols=len(Bs), figsize=(5*len(Bs),5))

for B, ax in zip(Bs, axs):
    ax.title.set_text(f"B = {B}")

    for model_type in accuracies_sorted_all.keys():
        sns.lineplot(np.arange(len(accuracies_sorted_all[model_type][str(B)]))/1000,
                     accuracies_sorted_all[model_type][str(B)], label=model_type, ax=ax,
                    linewidth=2)
    ax.set_xlabel("Number of points included [x1000]")
    ax.set_ylabel("Accuracy")
    ax.invert_xaxis()
    ax.legend(loc="lower right")
    ax.set_ylim([0.5,1.01])
fig.tight_layout()

fig.savefig(f"../../figures/{date_string}_{ex_name}_accuracy_curves.pdf")
fig.savefig(f"../../figures/{date_string}_{ex_name}_accuracy_curves.png")



