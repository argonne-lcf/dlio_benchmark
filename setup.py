from setuptools import setup, find_packages

setup(
    name='dlio_benchmark',
    version='0.0.1',
    include_package_data=True,
    install_requires=[
        'mpi4py',
        'numpy',
        'h5py',
        'pandas',
        'tensorflow >= 2.3.1',
        'pytorch >= 1.3.1'
    ],
    packages=find_packages(
        # All keyword arguments below are optional:
        where='src',  # '.' by default
        exclude=['tests'],  # empty by default
    ),
    entry_points={
        'console_scripts': [
            'cli-name = src.dlio_benchmark:main',
        ]
    }
)
