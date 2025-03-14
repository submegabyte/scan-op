from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_scan',
    ext_modules=[
        CUDAExtension('cuda_scan', ['scan.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
