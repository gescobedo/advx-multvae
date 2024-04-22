# Code to generate user latent embeddings to do plot analysis
python ../advx-multvae/src/gather.py --experiment "/results/<dataset_name>/vae--YYYY-MM-DD_HH-mm-SS" \
 --gpus "0,1" --n_parallel 6 --n_workers 1

