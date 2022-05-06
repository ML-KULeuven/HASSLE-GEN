import os
import sys
import numpy as np
from typing import Set, List, Optional, Tuple
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


Clause = Set[int]
Context = Set[int]
MaxSatModel = List[Tuple[Optional[float], Clause]]
Instance = np.ndarray  # A numpy array of 0s and 1s (False and True)
