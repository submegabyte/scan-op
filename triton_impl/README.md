# Triton implementations

Triton is an attempt to directly control GPU SRAM operations from the python layer itself while being platform independent (CUDA, ROCM).

These are naive implementations running in O(log L). Need to implement Blelloch's algorithm in Triton to bring it down.

Apparently Triton is being really buggy at the time, so we will be sticking to JAX at the HBM level.

https://github.com/sustcsonglin/mamba-triton/tree/master