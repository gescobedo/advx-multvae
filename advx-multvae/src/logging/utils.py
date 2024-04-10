import inspect
import contextlib
from tqdm import tqdm


@contextlib.contextmanager
def redirect_to_tqdm():
    """
    Redirects all print output through tqdm.write() to not mess up the
    output. For more information check out
    https://stackoverflow.com/a/42424890
    """

    # Store builtin print
    old_print = print

    def new_print(*args, **kwargs):
        # If tqdm.tqdm.write raises error, use builtin print
        try:
            tqdm.write(*args, **kwargs)
        except:
            old_print(*args, **kwargs)

    try:
        # Globaly replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print
