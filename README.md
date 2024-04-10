# AdvX-MultVAE
## Requirements 
First you should install  advx-multvae in a python environment. Please follow the detailed instructions in the folder `advx-multvae`.

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

