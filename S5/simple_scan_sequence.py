import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any, TypeVar

class SimpleScanSequence():
    def __init__(self, data, dim_L=1):
        # self.a = data
        self.data = data

        ## len(self.a.shape)
        # self.dim = self.a.dim()
        self.dim = self.data.dim()

        self.dim_L = dim_L
        # self.shape = self.a.shape
        self.shape = self.data.shape
        self.L = self.shape[self.dim_L]
    
    def __getitem__(self, i):
        ## i can be an integer, slice or a 1D array_like
        index = [slice(None)] * self.dim

        index[self.dim_L] = i
        # index[self.dim_L] = slice(i, i+1)

        # a = self.a[index]
        data = self.data[index]

        array_like = Union[torch.Tensor, list]
        if isinstance(i, slice) or isinstance(i, array_like):
            return self.__class__(data, self.dim_L)
        else:
            # return a
            return data
    
    def __setitem__(self, i, data):
        index = [slice(None)] * self.dim

        index[self.dim_L] = i
        # index[self.dim_L] = slice(i, i+1)

        # self.a[index] = data
        self.data[index] = data
    
    @classmethod
    def from_identity(cls, identity, L, dim_L=1):
        # shape = identity.shape[:dim_L] + (L,) + identity.shape[dim_L:]
        # dim = len(shape)
        dim = len(identity.shape) + 1

        r = [1,] * dim
        r[dim_L] = L
        data = identity.unsqueeze(dim_L).repeat(*r)
        return cls(data, dim_L=dim_L)

    def clone(self):
        data = self.data.clone()
        return self.__class__(data, dim_L=self.dim_L)

if __name__ == "__main__":

    B, L, D, N = 8, 512, 16, 8

    ## a: B, L, D, N
    ## h: B, L, D, N
    ## b: B, L, D, N
    ## x: B, L, D

    ##########

    # a = torch.randn(B, L, D, N)

    # c = SimpleScanSequence(a)

    # print(c[10])

    # c[10] = torch.ones(B, D, N)
    # print(c[10])

    ############

    identity = torch.ones(B, D, N)

    c = SimpleScanSequence.from_identity(identity, L)
    print(c.shape)
    # print(c.a)
    print(c[34:412].shape)
    print(c[[66,67,99,102,333]].shape)