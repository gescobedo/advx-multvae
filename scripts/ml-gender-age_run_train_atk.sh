conda activate recbole-bert

python ../cg_branch/src/train_and_attack.py --algorithm "vae" \
 --config ../configs/ml-gender-age.yaml \
 --atk_config ../configs/ml-gender-age-atk.yaml \
 --dataset ml-1m --gpus "0,1,2,3" --n_parallel 7 
