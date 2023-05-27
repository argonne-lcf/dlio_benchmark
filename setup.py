from setuptools import setup, find_packages
from glob import glob
configs = glob('dlio_benchmark/configs/**/*', recursive=True)
print(configs)
setup(
    name='dlio_benchmark',
    version='0.0.1',
    packages=find_packages(include=['dlio_benchmark', 'dlio_benchmark.*']),
    package_data={'dlio_benchmark/configs': configs},
    dependency_links=[
        'https://download.pytorch.org/whl/cpu',
        'https://developer.download.nvidia.com/compute/redist'
    ],
    install_requires=[
        'mpi4py',
        'numpy',
        'h5py',
        'pandas',
        'hydra-core == 1.2.0',
        'tensorflow == 2.11',
        'torch == 1.13.0',
        'torchaudio == 0.13.0',
        'torchvision == 0.14.0',
        'nvidia-dali-cuda110'
    ],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dlio_benchmark = dlio_benchmark.benchmark:main',
        ]
    }
)
