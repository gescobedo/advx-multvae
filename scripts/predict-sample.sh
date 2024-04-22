#script to obtain test results for test slices of advx-multvae
python ../advx-multvae/src/test.py --experiment "/results/<dataset_name>/vae--YYYY-MM-DD_HH-mm-SS" \
 --gpus "0,1" --n_parallel 6 --n_workers 1

