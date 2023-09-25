from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fw_cuda',
    ext_modules=[
        CUDAExtension('fw_cuda', [
            'fw_cuda.cpp',
            'fw_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
