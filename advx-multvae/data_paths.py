import os
import platform

# for multiple developers, create mapping of our computer names to data locations
# you can figure out your computer name by
# >>> import platform
# >>> platform.node()
_base_local_dataset_path_map = {
    "DESKTOP-RFBOALR": r"D:\OneDrive\Studium\04_Arbeit\Sommer2023\datasets",
    "AdminPC": r"E:\Studium\OneDrive\Studium\04_Arbeit\Sommer2023\datasets"
}

_base_local_results_path_map = {
    "DESKTOP-RFBOALR": r"D:\tmp\results",
}

# as all developers work on the same servers, instead of mapping computer names to locations,
# we map usernames to locations. You can list your username by
# >>> import os
# >>> os.getlogin()
_base_server_dataset_path_map = {
    "christian": "/share/hel/home/christian/datasets",
    "gustavoe": "/share/hel/datasets/lfm-sessions-sample"
}
_base_server_results_path_map = {
    "christian": "/home/christian/results/adv-research",
    "gustavoe": "/home/gustavoe/pers-bias/results-paper-balanced/"
    
}

relative_data_paths = {
    "lfm-small": os.path.join("lfm", "user_gte_10_item_gte_10_gender_age_loc_10000"),
    "lfm-big": os.path.join("lfm", "user_gte_10_item_gte_10_gender_age_loc_100000"),
    "lfm-100k-": os.path.join("lfm-100k", "user_gte_10_item_gte_10_gender_age_loc_100000"),
    "lfm-100k-norescale": os.path.join("lfm-100k-attr", "user_gte_10_item_gte_10_100000_norescale"),
    "lfm-100k": os.path.join("lfm-100k-attr", "user_gte_10_item_gte_10_age_100000"),

    "ml-1m": os.path.join("ml-1m", "user_gte_5_item_gte_5"),
    "ml-1m-norescale": os.path.join("ml-1m", "user_gte_5_item_gte_5"),
    "ml-100k": os.path.join("ml-100k", "user_gte_5_item_gte_5")
}

SUPPORTED_DATASETS = list(relative_data_paths.keys())
SUPPORTED_ALGORITHMS = ("vae", "mf")


def _is_running_on_server():
    return platform.node().startswith("rechenknecht")


def get_data_path(key):
    # determine whether we are running on server
    if _is_running_on_server():
        username = os.getlogin()
        if username not in _base_server_dataset_path_map:
            raise KeyError(f"No dataset location found for user '{username}' on server. "
                           f"Please extend '_base_server_dataset_path_map' in 'data_paths.py'.")
        path = os.path.join(_base_server_dataset_path_map[username], relative_data_paths[key])
    else:
        computer_name = platform.node()
        if computer_name not in _base_local_dataset_path_map:
            raise KeyError(f"No dataset location found on computer '{computer_name}'. "
                           f"Please extend '_base_local_dataset_path_map' in 'data_paths.py'.")
        path = os.path.join(_base_local_dataset_path_map[computer_name], relative_data_paths[key])

    return path


def get_storage_path():
    # determine whether we are running on server
    if _is_running_on_server():
        username = os.getlogin()
        if username not in _base_server_results_path_map:
            raise KeyError(f"No results location found for user '{username}' on server. "
                           f"Please extend '_base_server_results_path_map' in 'data_paths.py'.")
        path = _base_server_results_path_map[username]
    else:
        computer_name = platform.node()
        if computer_name not in _base_local_results_path_map:
            raise KeyError(f"No results location found on computer '{computer_name}'. "
                           f"Please extend '_base_local_results_path_map' in 'data_paths.py'.")
        path = _base_local_results_path_map[computer_name]
    return path
