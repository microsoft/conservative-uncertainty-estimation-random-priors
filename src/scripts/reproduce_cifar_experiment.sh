#!/bin/bash

for i in {1..10}
do

	python train_uncertainties.py --learning_rate 1e-4 --optimizer Adam --num_epochs 200 --init_scaling 2.0 --output_weight 1.0 --ensemble_size 1 --gp_weight 0. --batch_size 64 --data_set cifar10 --excluded_classes 0,1,3,4,8,9 --beta 1. --dropout_regularizer 1e-2 --ex_name RP --model_type uncertainty_ensemble

	python train_uncertainties.py --learning_rate 1e-4 --optimizer Adam --num_epochs 200 --init_scaling 2.0 --output_weight 1.0 --ensemble_size 1 --gp_weight 0. --batch_size 64 --data_set cifar10 --excluded_classes 0,1,3,4,8,9 --beta 1. --dropout_regularizer 1e-2 --ex_name DE --model_type deep_ensemble --adv_eps 0.

	python train_uncertainties.py --learning_rate 1e-4 --optimizer Adam --num_epochs 200 --init_scaling 2.0 --output_weight 1.0 --ensemble_size 1 --gp_weight 0. --batch_size 64 --data_set cifar10 --excluded_classes 0,1,3,4,8,9 --beta 1. --dropout_regularizer 1e-2 --ex_name DEAT --model_type deep_ensemble --adv_eps 0.1

	python train_uncertainties.py --learning_rate 1e-4 --optimizer Adam --num_epochs 200 --init_scaling 2.0 --output_weight 1.0 --ensemble_size 1 --gp_weight 0. --batch_size 64 --data_set cifar10 --excluded_classes 0,1,3,4,8,9 --beta 1. --dropout_regularizer 1e-2 --ex_name DR --model_type dropout

done

sleep 60

python reproduce_cifar_figure.py
