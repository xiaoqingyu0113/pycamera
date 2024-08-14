from setuptools import setup, find_packages

setup(
    name='pycamera',
    version='0.0.0',
    description='python library for camera projection and calibration',
    author='Qingyu Xiao',
    author_email='xiaoqingyu0113@gmail.com',
    url='https://github.com/qxiao33/pycamera',
    packages=find_packages(include=['pycamera', 'pycamera.*']),

    install_requires=[
        'opencv-python',
        'numpy'
    ],
    extras_require={'plotting': ['matplotlib']},
)