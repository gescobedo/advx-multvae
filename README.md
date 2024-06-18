# AdvX-MultVAE - Simultaneous Unlearning of Multiple Protected User Attributes From Variational Autoencoder Recommenders Using Adversarial Training
Source code corresponding to the paper:
> Escobedo, G, Ganhör, C., Brandl, S., Augstein, M., and Schedl, M. Simultaneous Unlearning of Multiple Protected User Attributes From Variational Autoencoder Recommenders Using Adversarial Training, Proceedings of the 5th International Workshop on Algorithmic Bias in Search and Recommendation (BIAS @ SIGIR 2024), Washington D.C., USA, July 2024.

## Requirements 
First you should install  advx-multvae in a python environment. Please follow the detailed instructions in the folder `advx-multvae`.

After that the notebooks in `advx-multvae/notebooks/data-preparation/` should be executed to generate the datasets
,this will generate several files. The splitting process for training is done automatically, once the experiments start. Please customize the data urls for saving the resultant files and also include them in the `advx-multvae/data_paths.py` file.

## Experiment configuration:
The folder `configs` contains all the configuration of our experiments :

__Lfm-2b-100k__:
- ./lfm-gender-age-atk.yaml
- ./lfm-gender-age.yaml

__ml-1m__:
- ./ml-gender-age.yaml
- ./ml-gender-age-atk.yaml

## Running experiments
The execution of our experiments can be reproduced de by executing the files in the `scripts` folder :
```
> cd scripts
> conda activate <envnname>
> . ml-gender-age_run_train_atk.sh
> . lfm-gender-age_run_train_atk.sh  
```
These scripts will execute the train and attack phases
## Evaluation
In order to obtain the test results of the pre-trained recommendation models, in the file `predict_sample.sh` replace the `--experiment` parameter with the corresponding results folder:
```
> cd scripts
> . predict-sample.sh

```
## Tensorboard
You can also examine results by pointin your tensorboard logdir to your experiments folder
```
> tensorboard --logdir="/results/<dataset_name>/vae--YYYY-MM-DD_HH-mm-SS"
```
