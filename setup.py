from setuptools import setup, find_namespace_packages
from glob import glob
from distutils import util
configs = glob('dlio_benchmark/configs/**/*', recursive=True)
print(configs)
test_deps = [
    'pytest',
]
core_deps = [
 'mpi4py>=3.1.4',
 'numpy>=1.23.5',
 'h5py>=3.7.0',
 'pandas>=1.5.1',
 'psutil',
 'dlio_profiler_py==0.0.3'
]
x86_deps = [
 'hydra-core >= 1.2.0',
 'tensorflow >= 2.11',
 'torch >= 1.13.0',
 'torchaudio',
 'torchvision',
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
import pathlib
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")
setup(
    name='dlio_benchmark',
    version='2.0',
    description="An I/O benchmark for deep Learning applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/argonne-lcf/dlio_benchmark",
    author="Hariharan Devarajan (Hari)",
    email="mani.hariharan@gmail.com",
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        # Indicate who your project is intended for
        "Intended Audience :: HPC",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache 2.0 License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="deep learning, I/O, benchmark, NPZ, pytorch benchmark, tensorflow benchmark",
    project_urls={  # Optional
        "Bug Reports": "https://github.com/argonne-lcf/dlio_benchmark/issues",
        "Source": "https://github.com/argonne-lcf/dlio_benchmark",
    },
    # Main package definition
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
            'dlio_benchmark = dlio_benchmark.main:main',
            'dlio_postprocessor = dlio_benchmark.postprocessor:main',
        ]
    }
)
