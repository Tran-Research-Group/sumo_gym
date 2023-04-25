from typing import TypeAlias

import numpy as np
from numpy import ndarray

ObsDict: TypeAlias = dict[str, ndarray]
InfoDict: TypeAlias = dict[str, ndarray | bool | float | np.floating]
