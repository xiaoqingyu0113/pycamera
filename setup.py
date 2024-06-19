from setuptools import setup, find_packages

setup(
    name='lfg',
    version='0.0.0',
    description='learnable factor graph using PyTorch',
    author='Qingyu Xiao',
    author_email='xiaoqingyu0113@gmail.com',
    url='https://github.com/xiaoqingyu0113/lfg',
    packages=find_packages(include=['lfg', 'lfg.*']),
    # package_data={
    #     'mcf4pingpong': ['darknet/*'],
    # },
    install_requires=[
        'PyYAML',
        'numpy',
        'matplotlib',
        'tqdm',
        'hydra-core',
        'omegaconf',
        'numba'
    ],
    extras_require={'plotting': ['matplotlib']},
)