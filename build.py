import setuptools
from setuptools import find_packages


setuptools.setup(    
    name='MCTensor',
    version='0.1',
    description='MCTensor: A High-Precision Deep Learning Library with Multi-Component Floating-Point',
    author='Tao Yu, Wentao Guo, Jianan Canal Li, Tiancheng Yuan, Christopher De Sa',
    url='https://github.com/ydtydr/MCTensor',
    project_urls={
        "Bug Tracker": "https://github.com/ydtydr/MCTensor",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
)