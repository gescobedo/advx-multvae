import os
import platform

# for multiple developers, create mapping of our computer names to data locations
# you can figure out your computer name by
# >>> import platform
# >>> platform.node()

_base_local_dataset_path_map = {
    "user": "/home/user/datasets" #Should be changed to use it
}

_base_local_results_path_map = {
   "user":"/home/user/results", #Should be changed to use it
}

#Datasets should be in /home/user/datasets Should be changed to use it
relative_data_paths = {
    "lfm-100k": os.path.join("lfm-100k", "user_gte_10_item_gte_10_age_100000"),
    "ml-1m": os.path.join("ml-1m", "user_gte_5_item_gte_5"),
  }

SUPPORTED_DATASETS = list(relative_data_paths.keys())
SUPPORTED_ALGORITHMS = ("vae")


def get_data_path(key):
    # determine whether we are running on server
    computer_name = platform.node()
    if computer_name not in _base_local_dataset_path_map:
        raise KeyError(f"No dataset location found on computer '{computer_name}'. "
                        f"Please extend '_base_local_dataset_path_map' in 'data_paths.py'.")
    path = os.path.join(_base_local_dataset_path_map[computer_name], relative_data_paths[key])

    return path


def get_storage_path():
    # determine whether we are running on server 
    computer_name = platform.node()
    if computer_name not in _base_local_results_path_map:
        raise KeyError(f"No results location found on computer '{computer_name}'. "
                        f"Please extend '_base_local_results_path_map' in 'data_paths.py'.")
    path = _base_local_results_path_map[computer_name]
    return path
