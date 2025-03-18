import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any, TypeVar

from simple_scan_sequence import SimpleScanSequence
from selective_scan_sequence import SelectiveScanSequence
from utils import *

## for selective scan
def add_op(a, b):
    return a[0] + b[0], a[1] + b[1]

device = "cuda" if torch.cuda.is_available() else "cpu"

class ScanSequence(SimpleScanSequence):
# class ScanSequence(SelectiveScanSequence):
    def naive_scan(self, op, identity):
        result = ScanSequence.from_identity(identity, self.L, dim_L=self.dim_L)

        state = identity
        for i in range(L):
            state = op(state, self[i])
            result[i] = state
        
        return result
    
    def blelloch_scan(self, op, identity):
        ## padding
        L_padded = padded(self.L)
        logL_padded = int(math.log2(L_padded))

        result = ScanSequence.from_identity(identity, L_padded, dim_L=self.dim_L)
        result[:self.L] = self.data

        ## up-sweep in O(logL_padded) time
        for depth in range(logL_padded):
            step = 2 ** (depth+1)

            ## Create indices for the operation
            indices = torch.arange(0, L_padded, step)
            if indices.numel() > 0:
                left_indices = indices + step//2 - 1
                right_indices = indices + step - 1

                ## Ensure indices are within bounds
                mask = right_indices < L_padded
                left_indices = left_indices[mask]
                right_indices = right_indices[mask]

                ## update values
                result[right_indices] = op(result[right_indices].data, result[left_indices].data)
        
                # print(depth, step, indices, left_indices, right_indices, result.data)
        
        ## Set the last element to identity element (for exclusive scan)
        up_sweep_sum = result[-1].clone()
        result[-1] = identity
        # print(f"up_sweep_sum: {up_sweep_sum}")
        # print(result.data)

        ## down-sweep in O(logL_padded) time
        for depth in range(logL_padded, 0, -1):
            step = 2 ** depth

            ## Create indices for the operation
            indices = torch.arange(0, L_padded, step)
            if indices.numel() > 0:
                left_indices = indices + step//2 - 1
                right_indices = indices + step - 1

                ## Ensure indices are within bounds
                mask = right_indices < L_padded
                left_indices = left_indices[mask]
                right_indices = right_indices[mask]

                ## swap and combine
                temp = result[left_indices].clone()
                result[left_indices] = result[right_indices].data
                result[right_indices] = op(result[right_indices].data, temp.data)

                # print(depth, step, indices, left_indices, right_indices, result.data)
        
        ## unpad and left shift
        # result = result[:self.L]
        temp = self.clone()
        temp[:self.L-1] = result[1:self.L].data
        temp[-1] = up_sweep_sum
        result = temp
        # print(result.data)
        return result

if __name__ == "__main__":
    B, L, D, N = 8, 512, 16, 8
    # B, L, D, N = 2, 4, 4, 3
    # B, L, D, N = 1, 512, 1, 1
    # L = 8

    op = operator.add
    # op = add_op

    # identity = torch.ones(B, D, N)
    identity = torch.zeros(B, D, N)
    # identity = torch.tensor(0)
    # identity = (torch.ones(B, D, N), torch.zeros(B, D, N))

    # a = ScanSequence.from_identity(identity, L)
    # data = torch.randn(B, L, D, N)
    data = torch.randint(-1000, 1000, (B, L, D, N))
    # data = torch.tensor([3, 1, 7, 0, 4, 1, 6, 3])
    a = ScanSequence(data, dim_L=1)
    # print(a.data)

    h_naive = a.naive_scan(op, identity)
    h_blelloch = a.blelloch_scan(op, identity)

    # h_naive = a.naive_scan(add_op, identity)
    # h_blelloch = a.blelloch_scan(add_op, identity)

    # print(h_naive.data)
    # print(h_blelloch.data)
    
    # print(h.data[0])
    # print(torch.all(h_naive.data[1] == h_blelloch.data[1]))
    print(torch.all(h_naive.data == h_blelloch.data))
    # print(h_naive.data - h_blelloch.data)
    # isclose_ = torch.isclose(h_naive.data, h_blelloch.data)
    # print(isclose_)
    # print(torch.all(isclose_))
        