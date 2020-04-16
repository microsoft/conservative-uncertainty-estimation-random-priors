"""
Script for training the uncertainty models on different data sets.

Copyright 2019
Vincent Fortuin
Microsoft Research Cambridge
"""

import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from absl import app
from absl import flags
import datetime
import uuid
import json
import shutil

from core import *
from torch_backend import *

import models
import model_helpers
import datasets

FLAGS = flags.FLAGS

sns.set(context="paper", style="whitegrid")

flags.DEFINE_integer("output_size", 512, "Size of the uncertainty networks' latent outputs.")
flags.DEFINE_float("learning_rate", 0.0001, "Maximum learning rate for the uncertainty training.")
flags.DEFINE_integer("num_epochs", 200, "Number of epochs for uncertainty training.")
flags.DEFINE_string("ex_name", "debug", "Name for the experiment.")
flags.DEFINE_float("init_scaling", 2.0, "Scaling factor for the parameters of the prior networks.")
flags.DEFINE_float("output_weight", 1.0, "Weight to scale the output of the uncertainty networks.")
flags.DEFINE_integer("seed", 0, "Seed for the RNG.")
flags.DEFINE_boolean("hyperopt", False, "Flag for optimizing the hyperparameters.")
flags.DEFINE_integer("ensemble_size", 1, "Number of uncertainty pairs in the ensemble.")
flags.DEFINE_float("gp_weight", 0., "Weighting parameter for the gradient penalty loss.")
flags.DEFINE_integer("batch_size", 512, "Batch size for the training.")
flags.DEFINE_integer("num_runs", 1, "Number of consecutive training runs.")
flags.DEFINE_list("excluded_classes", [], "Classes from the data that should be withheld for testing.")
flags.DEFINE_integer("num_excluded_classes", 0, "If set to a nonzero N, this overrides the excluded_classes argument and excludes the first N classes.")
flags.DEFINE_enum("model_type", "uncertainty_ensemble", ["uncertainty_ensemble", "deep_ensemble", "dropout"], "Type of model to train.")
flags.DEFINE_enum("data_set", "cifar10", ["cifar10", "cifar100", "svhn", "imagenet", "cifar10_reduced", "imagenet_reduced"], "Data set to train on.")
flags.DEFINE_enum("optimizer", "Adam", ["SGD", "Adam"], "Type of optimizer to use.")
flags.DEFINE_string("pretrained_model", None, "Filepath to a pretrained model. If specified,"
                    " this model is used for training and the model_type is ignored.")
flags.DEFINE_float("adv_eps", 0., "Epsilon parameter for the adversarial training in the deep ensemble.")
flags.DEFINE_float("beta", 1., "Tradeoff parameter between the mean standard deviation and the bootstrap bonus in the uncertainty ensemble.")
flags.DEFINE_float("dropout_regularizer", 1e-2, "Weight for the dropout regularizer.")
flags.DEFINE_float("reduced_size", 75, "Reduced size of the training set when training on cifar10_reduced or imagenet_reduced.")


def main(argv):
    del argv # unused
        
    if FLAGS.num_excluded_classes > 0:
        FLAGS.excluded_classes = np.arange(FLAGS.num_excluded_classes)
    
    FLAGS.excluded_classes = [int(cls) for cls in FLAGS.excluded_classes]
    
    # Get random hyperparameters if we want to do a hyperopt
    if FLAGS.hyperopt:
        hparams = model_helpers.get_hparams(FLAGS.seed)
        FLAGS.output_size = hparams["M"]
        FLAGS.learning_rate = hparams["lr"]
        FLAGS.init_scaling = hparams["init_scale"]
        FLAGS.output_weight = hparams["out_weight"]
    
    flags_dict = {flag : FLAGS[flag].value for flag in FLAGS}
    
    print(f"config: {flags_dict}")
    
    # Make a nice name for our experiment
    datestring = datetime.date.today().strftime("%y%m%d")
    run_id = uuid.uuid4().hex[:10]
    logdir = f"../../logs/{datestring}_{FLAGS.ex_name}_{run_id}"
    
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "models"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "results"), exist_ok=True)
    
    
    # Load data
    DATA_DIR = './data'
    
    if FLAGS.data_set == "cifar10":
        train_set, test_set, excluded_set = datasets.get_cifar10_data(data_dir=DATA_DIR,
                                                    exclude_classes=FLAGS.excluded_classes)
    elif FLAGS.data_set == "cifar100":
        train_set, test_set, excluded_set = datasets.get_cifar100_data(data_dir=DATA_DIR,
                                                    exclude_classes=FLAGS.excluded_classes)
    elif FLAGS.data_set == "svhn":
        train_set, test_set, excluded_set = datasets.get_svhn_data(data_dir=DATA_DIR,
                                                    exclude_classes=FLAGS.excluded_classes)
    elif FLAGS.data_set == "imagenet":
        train_set, test_set, excluded_set = datasets.get_imagenet_data(data_dir=DATA_DIR,
                                                    exclude_classes=FLAGS.excluded_classes)
    elif FLAGS.data_set == "cifar10_reduced":
        train_set, test_set, excluded_set = datasets.get_cifar10_reduced_data(data_dir=DATA_DIR,
                                                    exclude_classes=FLAGS.excluded_classes,
                                                        reduced_size=FLAGS.reduced_size)
    elif FLAGS.data_set == "imagenet_reduced":
        train_set, test_set, excluded_set = datasets.get_imagenet_reduced_data(data_dir=DATA_DIR,
                                                    exclude_classes=FLAGS.excluded_classes,
                                                        reduced_size=FLAGS.reduced_size)
    
    # Set training parameters
    epochs= FLAGS.num_epochs
    output_size = FLAGS.output_size
    init_scaling = FLAGS.init_scaling
    output_weight = FLAGS.output_weight
    ensemble_size = FLAGS.ensemble_size
    gp_weight = FLAGS.gp_weight
    batch_size = FLAGS.batch_size
    N_runs = FLAGS.num_runs
    data_size = len(train_set)
    
    lr_schedule = PiecewiseLinear([0, epochs//3, epochs], [0, FLAGS.learning_rate, 0])
    lr = lambda step: lr_schedule(step/len(train_batches))/batch_size
    
    
    train_batches = Batches(train_set, batch_size, shuffle=True, drop_last=True)    # without transforms
    test_batches = Batches(test_set, batch_size, shuffle=False, drop_last=False)
    excluded_batches = Batches(excluded_set, batch_size, shuffle=False, drop_last=False)    # currently unused
    
    # Train the model
    summaries = []
    for i in range(N_runs):
        print(f'Starting Run {i} at {localtime()}')
        if FLAGS.pretrained_model is not None:
            model = torch.load(FLAGS.pretrained_model)
        elif FLAGS.model_type == "uncertainty_ensemble":
            model = models.TorchUncertaintyEnsemble(ensemble_size=ensemble_size,
                                                    output_size=output_size,
                                                    init_scaling=init_scaling,
                                                    output_weight=output_weight,
                                                    gp_weight=gp_weight,
                                                    beta=FLAGS.beta).to(device).half()
        elif FLAGS.model_type == "deep_ensemble":
            model = models.TorchDeepEnsemble(ensemble_size=ensemble_size,
                                            output_size=output_size,
                                            output_weight=output_weight,
                                            adv_eps=FLAGS.adv_eps).to(device).half()
        elif FLAGS.model_type == "dropout":
            model = models.DropoutModel(bootstrap_size=ensemble_size,
                                       output_size=output_size,
                                       weight_regularizer=FLAGS.dropout_regularizer/data_size,
                                       dropout_regularizer=2./data_size).to(device).half()
        if FLAGS.optimizer == "SGD":
            opt = SGD(trainable_params(model), lr=lr, momentum=0.9, weight_decay=5e-4*batch_size, nesterov=True)
        elif FLAGS.optimizer == "Adam":
            opt = torch.optim.Adam(trainable_params(model), lr=FLAGS.learning_rate, weight_decay=5e-4*batch_size, eps=1e-4)
        try:
            summaries.append(train(model, opt, train_batches, test_batches, epochs, loggers=(TableLogger(),
                                                        FileLogger(os.path.join(logdir, "training_curve.tsv")))))
        except KeyboardInterrupt:
            print("Training run interrupted.")
            summaries = [{}]
            

    # Save the config we used
    with open(os.path.join(logdir, "results", "config.json"), "w") as outfile:
        outfile.write(json.dumps(flags_dict))
        
    # Save the trained model
    torch.save(model, os.path.join(logdir, "models", "uncertainty_model"))
    
    # Save a copy of this code
    shutil.copy2(sys.argv[0], logdir)
    
    # Compute the uncertainties on the train and test set
    uncertainties_train_all = []
    uncertainties_test_all = []
    uncertainties_excluded_all = []

    for batch in train_batches:
        uncertainties_train_all.extend(model(batch)['uncertainties'].cpu().detach().numpy())

    for batch in test_batches:
        uncertainties_test_all.extend(model(batch)['uncertainties'].cpu().detach().numpy())
        
    for batch in excluded_batches:
        uncertainties_excluded_all.extend(model(batch)['uncertainties'].cpu().detach().numpy())

    uncertainties_train_all = np.array(uncertainties_train_all, dtype=np.float32)
    uncertainties_test_all = np.array(uncertainties_test_all, dtype=np.float32)
    uncertainties_excluded_all = np.array(uncertainties_excluded_all, dtype=np.float32)
    
    # Add the computed uncertainties to our summaries and save them
    summaries = summaries[0]
    
    summaries['uncertainties train mean'] = np.mean(uncertainties_train_all).astype(np.float)
    summaries['uncertainties train std'] = np.std(uncertainties_train_all).astype(np.float)
    
    summaries['uncertainties test mean'] = np.mean(uncertainties_test_all).astype(np.float)
    summaries['uncertainties test std'] = np.std(uncertainties_test_all).astype(np.float)
    
    if len(FLAGS.excluded_classes) >= 1:
        summaries['uncertainties excluded mean'] = np.mean(uncertainties_excluded_all).astype(np.float)
        summaries['uncertainties excluded std'] = np.std(uncertainties_excluded_all).astype(np.float)
    else:
        summaries['uncertainties excluded mean'] = 0.
        summaries['uncertainties excluded std'] = 0.
    
    print(summaries)
    
    with open(os.path.join(logdir, "results", "summaries.json"), "w") as outfile:
        outfile.write(json.dumps(summaries))

    uncertainties_all = np.concatenate([uncertainties_train_all, uncertainties_test_all])


if __name__ == '__main__':
    app.run(main)