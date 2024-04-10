# Experiments

## Overview

- [Introduction](#introduction)
- [Config files](#config-files)
    - [Sample configuration for training a simple VAE](#sample-configuration-for-training-a-simple-vae)
    - [Sample configuration for training a simple VAE + adversarial network](#sample-configuration-for-training-a-simple-vae--adversarial-network)
    - [Remarks](#remarks)
    - [Grid search](#grid-search)
- [Usage of ```train_vae.py``` and ```train_and_attack.py```](#usage-of-train_vaepy-and-train_and_attackpy)
- [Usage of ```val_vae.py``` and ```attack_vae.py```](#usage-of-val_vaepy-and-attack_vaepy)
- [Final configurations used to report results in paper](#final-configurations-used-to-report-results-in-paper)

## Introduction

For experiments, different scripts are available in the ```src\``` directory:

- ```train_vae.py```: Trains a Variational Autoencoder (VAE), optionally with an adversarial network
- ```attack_vae.py```: Trains an attacker network on the latent space of a previously trained model
- ```train_and_attack.py```: Similar to ```train_vae.py```, but immediately attacks trained models afterwards
- ```val_vae.py```: Evaluates / calculates metrics for previously trained models

**Note:** scripts should be executed in the ```/src``` directory, as paths for e.g. storing results are defined
relatively to it.

The scripts support additional parameters that must be specified. Some of them are optional.

## Config files - TODO: Update

As there are a lot of parameters that may be specified (and tuned), config files
are required for training models.
The configuration for an experiment must be specified in a ```.json``` file. To ease the readability,
the configuration is separated into groups, where each group should represent a certain module that
is relevant for training (e.g., VAE architecture, optimizer parameters, ...).  
Note: The group names have to follow the pattern ```<group>_params```.

### Sample configuration for training a simple VAE

```
TODO: Update to new config
```

### Sample configuration for training a simple VAE + adversarial network

```
TODO
```

### Remarks

Additionally to the before mentioned parameters, for each module of the model, different optimizer
parameters may be used. Similar to the ```opt_params``` group, the following additional groups may be specified:

- ```enc_opt_params```
- ```dec_opt_params```
- ```adv_opt_params```

In case they are not specified, the parameters default to ```opt_params```.

For a complete list of all the available parameters, check out the different configuration files lying around in
[ml_configs](/configs/ml) as well as in [lfm_configs](/configs/lfm) and search for their behaviour in the source code.

### Grid search

Hyper-parameter search is an important part in order to determine the best model configuration.
Therefore, the configuration files also supports a basic grid search. To perform grid search, you
need to pack the parameters to search into a list and move them into a separate group that is
named ```<group>_search_params```.

Example for grid search on ```p_dims``` and ```input_dropout```:

```
TODO:
```

### Usage of ```train_vae.py``` and ```train_and_attack.py```

```
usage:
    train_vae.py / train_and_attack.py [-h]
    --experiment_type {standard,up_sample} --config CONFIG [--ncores NCORES] [--n_parallel n_parallel]
    [--dataset {lfm2b,movielens}] [--gpus GPUS] [--n_folds {1,2,3,4,5}]  [--store_best STORE_BEST]
    [--store_every 0..99]
    
options:
    -h, --help              Show this help message and exit
    
    --experiment_type       The type of experiment that should be performed.
                            Choices:
                                standard ... uses the dataset splits to train the recommender system
                                up_sample ... upsamples (oversamples) the training set
                                                            
    --config CONFIG         The configuration to use when training a model.
                            Supports gridsearch (for more, see in `/CONFIG.md`)
                                                    
    --gpus GPUS             The GPUs to use for training, use e.g., '0,2' to run on GPU '0' and '2'
    --n_folds                The number of folds the models should be evaluated on. (default=5)
                            Choices: {1,2,3,4,5} 
                            
    --ncores NCORES         The number of cores/workers that each dataloader should use
    --n_parallel n_parallel   The number of processes that should be run in parallel on each device, (default=1)
                                                       
    --dataset               The dataset to train / run the models on.      
                            Choices:
                                lfm2b ... (default) uses the LFM2b-demo dataset
                                movielens ... usees the MovieLens-1m dataset
                                                                                            
    --store_best STORE_BEST Whether the best models should be stored,
                            i.e., whether early stopping should be performed.
                            
    --store_every           After which number of epochs the model should be stored, 
                            0 to deactivate this feature, (default=0)
                            Choices: [0 .. 99]
```

### Usage of ```val_vae.py``` and ```attack_vae.py```

```             
usage: val_vae.py [-h] [--run RUN] [--experiment EXPERIMENT] [--n_folds {1,2,3,4,5}]
[--gpus GPUS] [--ncores NCORES] [--split {val,test}] [--use_tensorboard USE_TENSORBOARD]
               
usage: attack_vae.py [-h] [--run RUN] [--experiment EXPERIMENT] [--config CONFIG] [--n_folds {1,2,3,4,5}]
[--gpus GPUS] [--ncores NCORES] [--split {val,test}] [--use_tensorboard USE_TENSORBOARD]

options:
    -h, --help                          Show this help message and exit
    --run RUN                           The path to a run that should be attacked / validated.
    --experiment EXPERIMENT             The path to an experiment, i.e., collection of multiple runs, 
                                        where each one should be attacked / validated
    --n_folds {1,2,3,4,5}                The number of folds the models should be evaluated on.
    --gpus GPUS                         The gpus to run the models on, use e.g., '0,2' to run on GPU '0' and '2'
    --ncores NCORES                     The number of cores that each dataloader should use
    --split {val,test}                  The split to attack / validate upon.
    --use_tensorboard USE_TENSORBOARD   Whether results and additional information should be logged via tensorboard 
                                        (in addition to writing to files)

options only for attack_vae.py:
    --config CONFIG                     The config file for the attacker network.                            
```

## Final configurations used to report results in paper

For the final results that are reported in the paper, we used the following configurations:

- **LFM2b** dataset: [/configs/lfm/final_adv.json](/configs/lfm/final_adv.json)
- **MovieLens-1M** dataset: [/configs/ml/final_adv.json](/configs/ml/final_adv.json)
