from pathlib import Path
import argparse
import os

from data_paths import SUPPORTED_DATASETS, SUPPORTED_ALGORITHMS
from src.utils.helper import pretty_print
from src.utils.input_options import input_options

from collections import namedtuple


def _add_input_option(parser, option):
    assert input_options.get(option), f"Option '{option}' not available, needs to be introduced first."
    parser.add_argument("--" + option, **input_options[option])


def _get_devices(args, options: list):
    check_gpus = "gpus" in options and args.gpus != ""
    if check_gpus:
        print("=" * 60)
        print(f"Setting 'CUDA_VISIBLE_DEVICES' to '{args.gpus}'. ")
        print("=" * 60)

        # Adjust environment variable, so we can use all visible GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Import torch after setting environment variable to ensure that environment variable
    # takes affect
    import torch

    if check_gpus and torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        devices = [torch.device(f"cuda:{i}") for i in
                   range(torch.cuda.device_count())] if n_devices > 1 else [torch.device("cuda")]
        devices *= args.n_parallel if "n_parallel" in options else 1
    else:
        devices = [torch.device("cpu")]
    return devices


def _populate_run_dict(run_dir, d, model_pattern="*.pt*"):
    config_file = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_file):
        return

    # use pathlib.glob as glob.glob doesn't seem to work with our run names
    model_files = [str(f) for f in Path(run_dir).glob(model_pattern)]
    model_names = [os.path.splitext(os.path.basename(f))[0] for f in model_files]

    if len(model_files) > 0:
        d[run_dir] = {"config_file": config_file, "models": list(zip(model_names, model_files))}
    else:
        print(f"Run in dir '{run_dir}' not usable as it doesn't contain a model file (.pt | .pth)")


def _get_runs(args, options: list):
    print(options)
    if "run" not in options and "experiment" not in options:
        return None

    if "run" in options and "experiment" in options:
        if (args.run is None) == (args.experiment is None):
            raise AttributeError("Either run (x)or experiment has to be specified")

    model_pattern = "*" if "model_pattern" not in options else args.model_pattern

    run_dict = dict()
    if args.run is not None:
        _populate_run_dict(args.run, run_dict, args.model_pattern)
    elif args.experiment is not None:
        # determine directories that contain runs
       
        search_str = os.path.join( "**","train", "**", "events.out.*")  # TOOD: This hardcoded "train" kinda sux
       
        print(f"Getting all files that match glob regex '{search_str}' in '{args.experiment}'")
        runs = list(Path(args.experiment).rglob(search_str))
      
        for run in runs:
            _populate_run_dict(str(run.parent), run_dict, model_pattern)
    return run_dict


def _process_input(args, options: list):
    processed = {
        "devices": _get_devices(args, options)
    }

    if "config" in options:
        if not os.path.isfile(args.config):
            raise ValueError(f"Config file '{args.config}' does not exist.")
        processed["config"] = args.config
    print("to runs read")
    if rd := _get_runs(args, options):
        print("runs read")
        processed["run_dict"] = rd
        for ds in SUPPORTED_DATASETS:
            if f"{os.path.sep}{ds}{os.path.sep}" in (args.run or args.experiment):
                processed["dataset"] = ds
                break
        else:
            raise ValueError("Could not determine the name of the dataset, please ensure that you "
                             "did not change the directory structure!")

        for algo in SUPPORTED_ALGORITHMS:
            if f"{os.path.sep}{algo}--" in (args.run or args.experiment):
                processed["algorithm"] = algo
                break
        else:
            raise ValueError("Could not determine the algorithm, please ensure that you "
                             "did not change the directory structure!")

    # Take the options as is which do not require special processing or may still be nice to know as is
    not_processed_options = set(options) - {"gpus", "run", "experiment", "model_pattern"}
    for opt in not_processed_options:
        processed[opt] = args.__dict__[opt]

    # return aggregated, preprocessed input
    return processed


def _summarize_input(run_type, processed):
    print("=" * 60)
    print(f"Starting {run_type}!")
    print("Using input config:")
    tmp = processed.copy()
    tmp.pop("devices")
    pretty_print(tmp)
    print("devices:", processed["devices"])
    print("=" * 60)


def parse_input(run_type: str, options, access_as_properties=True):
    parser = argparse.ArgumentParser()

    # add options 
    for opt in options:
        _add_input_option(parser, opt)
    args = parser.parse_args()

    processed = _process_input(args, options)
    _summarize_input(run_type, processed)

    # accessing the input as properties rather than by indexing a dict may be
    # nicer for some use-cases
    if access_as_properties:
        keys, values = zip(*processed.items())
        processed = namedtuple("InputValues", field_names=keys, defaults=values)()

    return processed
