import multiprocessing

MAX_FOLDS = 5

EXP_SEED = 42  # Seed for algorithms that rely on random initialization
N_WORKERS = min(1, multiprocessing.cpu_count())  # prevent excessive use of workers if no more cpu cores are available

ACCEPTABLE_MODEL_DIRS = ("train",)
