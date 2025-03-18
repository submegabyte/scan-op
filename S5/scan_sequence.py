import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any, TypeVar

from simple_scan_sequence import SimpleScanSequence
from selective_scan_sequence import SelectiveScanSequence

## for selective scan
def add_op(a, b):
    return a[0] + b[0], a[1] + b[1]

# class ScanSequence(SimpleScanSequence):
class ScanSequence(SelectiveScanSequence):
    def naive_scan(self, op, identity, dim_L=1):
        result = ScanSequence.from_identity(identity, self.L)

        state = identity
        for i in range(L):
            state = op(state, self[i])
            result[i] = state
        
        return result

if __name__ == "__main__":
    B, L, D, N = 8, 512, 16, 8

    # identity = torch.ones(B, D, N)
    # identity = torch.zeros(B, D, N)
    identity = (torch.ones(B, D, N), torch.zeros(B, D, N))

    a = ScanSequence.from_identity(identity, L)

    # h = a.naive_scan(torch.add, identity)
    h = a.naive_scan(add_op, identity)

    print(h.data[0])
        