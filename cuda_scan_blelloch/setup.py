from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_blelloch_scan',
    ext_modules=[
        CUDAExtension('cuda_blelloch_scan', ['scan.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
