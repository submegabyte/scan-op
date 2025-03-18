import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any, TypeVar

## pad to the higher power of 2
## 1 -> 1
## 2 -> 2
## 3, 4 -> 4
## 5, 6, 7, 8 -> 8
def padded(L):
    logL = math.log2(L)
    logL_padded = math.ceil(logL)
    L_padded = 2 ** logL_padded
    return L_padded