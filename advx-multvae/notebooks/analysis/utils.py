import re
import os
import glob
import numpy as np
import pandas as pd
import pickle as pkl

import zipfile
from collections import defaultdict
from tempfile import TemporaryDirectory
from tbparser.summary_reader import SummaryReader

from pathlib import Path

def get_params(s):
    rc = re.compile("'(.*?)'=(\[{0,1}[\-0-9\., a-zA-Z]*?\]{0,1})[,\}]")
    params = {}
    for k, v in rc.findall(s):
        try:
            params[k] = float(v)

            try:
                params[k] = int(v)
            except:
                pass
        except:
            params[k] = v
    return params
    

def get_values(data, tag, run):
    # retrieve data and ensure that it is ordered
    return np.array([item.value for item in sorted(data[tag][run], key=lambda x: x.step)])


def get_step_values(data, tag, run):
    # retrieve data and ensure that it is ordered
    return np.array([(item.step, item.value) for item in sorted(data[tag][run], key=lambda x: x.step)])
    
def load_csvs(path, run_names):
    results = defaultdict(lambda: dict())
    
    for run in run_names:
        files = (Path(path) / run).glob("*.csv")
        
        for f in files:
            file_name = os.path.basename(f)
            data_name = file_name[:-len("_adv_scores.csv")] if "_adv_scores.csv" in file_name else file_name[:-len(".csv")]
            results[run][data_name] = pd.read_csv(f, sep=";")
        
    return dict(results)

def load_pkls(path, run_names):
    results = defaultdict(lambda: dict())
    
    for run in run_names:
        files = (Path(path) / run).glob("*.pkl")
        
        for f in files:
            file_name = os.path.basename(f)
            data_name = file_name[:-len(".csv")]
            
            with open(f, "rb") as fh:
                results[run][data_name] = pkl.load(fh)
            
    return dict(results)

def load_features(path, run_names):
    results = defaultdict(lambda: dict())
    for run in run_names:        
        results_files = (Path(path) / run).glob("*.pkl")
        
        for f in results_files:
            r = re.search("epoch_(\d*)_results", os.path.basename(f))
            epoch = int(r[1]) if r else os.path.basename(f)[:-len("_results.pkl")]
            
            file_name = os.path.basename(f)
            data_name = file_name[:-len("_results.pkl")] if "_results.pkl" in file_name else file_name[:-len(".pkl")]
            with open(f, "rb") as fh:
                results[run][data_name] = pkl.load(fh)

    return dict(results)


def load_results(path, folds=[0], include_tensorboard=True, include_csvs=False, include_pkls=False):
    results, results_retr, tb_data, csv_results, pkl_results, csv_results_atk = {}, {}, {}, {}, {}, {}
    for fold in folds:
        print("Using data of fold", fold)
        dir_path = os.path.join(path, str(fold))
        
        tb_data[fold] = None
        if include_tensorboard:
            # Load tensorboard data
            print("Loading tensorboard data...")
            sr = SummaryReader(dir_path)
            tb_data[fold] = sr.load_scalar_data()

        run_dir = os.path.join(dir_path, "vae")
        run_names = [os.path.relpath(item, run_dir) for item in glob.glob(os.path.join(run_dir, "*")) if os.path.isdir(item)]

        atk_dir = os.path.join(dir_path, "atk")
        atk_names = [os.path.relpath(item, atk_dir) for item in glob.glob(os.path.join(atk_dir, "*")) if os.path.isdir(item)]

        results[fold] = load_features(os.path.join(dir_path, "vae_test_features"), run_names)
        
        csv_results_atk[fold] = None
        if include_csvs:
            csv_results_atk[fold] = load_csvs(os.path.join(dir_path, "atk"), atk_names)
            
        csv_results[fold] = None
        if include_csvs:
            csv_results[fold] = load_csvs(os.path.join(dir_path, "vae_test_eval"), run_names)
            
        pkl_results[fold] = None
        if include_pkls:
            pkl_results[fold] = load_pkls(os.path.join(dir_path, "vae_test_eval"), run_names)
        
        path_retrained = os.path.join(dir_path, "retrain_test_features")
        results_retr[fold] = None
        if os.path.exists(path_retrained):
            results_retr[fold] = load_features(path_retrained, run_names)
    
    return results, results_retr, tb_data, csv_results, pkl_results, csv_results_atk

 
def load_results_zip(zip_path, folds=[0], include_tensorboard=True, include_csvs=False, include_pkls=False):
    print("Loading zip '%s'" % zip_path)
    with TemporaryDirectory() as tmp_dir:
        print("Extracting files to temporary directory:", tmp_dir)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        return load_results(tmp_dir, folds, include_tensorboard, include_csvs, include_pkls)


def smooth_values(x, smoothing=0.5):
    return np.array(pd.DataFrame(x).ewm(alpha=1 - smoothing).mean()[0].tolist())

