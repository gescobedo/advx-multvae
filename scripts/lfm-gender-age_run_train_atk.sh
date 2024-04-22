conda activate recbole-bert

python ../cg_branch/src/train_and_attack.py --algorithm "vae" \
 --config ../configs/lfm-gender-age.yaml \
 --atk_config ../configs/lfm-gender-age-atk.yaml \
 --dataset lfm-100k --gpus "0,1,2,3" --n_parallel 6 
