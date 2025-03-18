import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any, TypeVar

class SelectiveScanSequence():
    def __init__(self, data, dim_L=1):
        # self.a = data[0]
        # self.b = data[1]
        self.data = data[0], data[1]

        ## len(self.a.shape)
        # self.dim = self.a.dim()
        self.dim = self.data[0].dim()

        self.dim_L = dim_L
        # self.shape = self.a.shape
        self.shape = self.data[0].shape
        self.L = self.shape[self.dim_L]
    
    def __getitem__(self, i):
        index = [slice(None)] * self.dim

        index[self.dim_L] = i
        # index[self.dim_L] = slice(i, i+1)

        # a = self.a[index]
        # b = self.b[index]
        a = self.data[0][index]
        b = self.data[1][index]

        return a, b
    
    def __setitem__(self, i, data):
        index = [slice(None)] * self.dim

        index[self.dim_L] = i
        # index[self.dim_L] = slice(i, i+1)

        # self.a[index] = data[0]
        # self.b[index] = data[1]
        self.data[0][index] = data[0]
        self.data[1][index] = data[1]
    
    @classmethod
    def from_identity(cls, identity, L, dim_L=1):
        ## identity[0].shape == identity[1].shape == B, D, N
        # shape = identity[0].shape[:dim_L] + (L,) + identity[0].shape[dim_L:]
        # dim = len(shape)
        dim = len(identity[0].shape) + 1

        r = [1,] * dim
        r[dim_L] = L
        a = identity[0].unsqueeze(dim_L).repeat(*r)
        b = identity[1].unsqueeze(dim_L).repeat(*r)
        data = (a, b)
        return cls(data, dim_L=dim_L)

if __name__ == "__main__":

    B, L, D, N = 8, 512, 16, 8

    ## a: B, L, D, N
    ## h: B, L, D, N
    ## b: B, L, D, N
    ## x: B, L, D

    #############

    # a = torch.randn(B, L, D, N)
    # bx = torch.randn_like(a)

    # c = SelectiveScanSequence((a, bx))

    # print(c[10])

    # c[10] = (torch.ones(B, D, N), torch.ones(B, D, N))
    # print(c[10])

    ###############

    identity = (torch.ones(B, D, N), torch.zeros(B, D, N))

    c = SelectiveScanSequence.from_identity(identity, L)
    print(c.a.shape, c.b.shape)
    # print(c.a)