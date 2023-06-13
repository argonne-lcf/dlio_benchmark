from setuptools import setup, find_namespace_packages
from glob import glob
from distutils import util
configs = glob('dlio_benchmark/configs/**/*', recursive=True)
print(configs)
test_deps = [
    'pytest',
]
core_deps = [
 'mpi4py',
 'numpy',
 'h5py',
 'pandas'
]
x86_deps = [
 'hydra-core == 1.2.0',
 'tensorflow == 2.11',
 'torch == 1.13.0',
 'torchaudio == 0.13.0',
 'torchvision == 0.14.0',
 'nvidia-dali-cuda110' 
]
ppc_deps = [
 'hydra-core @ git+https://github.com/facebookresearch/hydra.git@v1.3.2#egg=hydra-core'
]
deps = core_deps
if "ppc" in util.get_platform():
  deps.extend(ppc_deps)
else:
  deps.extend(x86_deps)
print(deps)
extras = {
    'test': test_deps,
}
setup(
    name='dlio_benchmark',
    version='0.0.1',
    packages=find_namespace_packages(where="."),
    package_dir={"dlio_benchmark": "dlio_benchmark"},
    package_data={'dlio_benchmark.configs': ['*.yaml'],
                  'dlio_benchmark.configs.hydra.help': ['*.yaml'],
                  'dlio_benchmark.configs.hydra.job_logging': ['*.yaml'],
                  'dlio_benchmark.configs.workload': ['*.yaml'],
                  },
    dependency_links=[
        'https://download.pytorch.org/whl/cpu',
        'https://developer.download.nvidia.com/compute/redist'
    ],
    install_requires=deps,
    tests_require=test_deps,
    extras_require=extras,
    entry_points={
        'console_scripts': [
            'dlio_benchmark = dlio_benchmark.benchmark:main',
            'dlio_postprocesser = dlio_benchmark.dlio_postprocesser:main',
        ]
    }
)
