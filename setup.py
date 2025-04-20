from setuptools import setup, find_packages

setup(
    name="janc",
    version="0.1.0",
    package_dir={"janc": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.7',
    ],
    entry_points={
        'console_scripts': [
            'cfs-amr-run = main:solution',
        ],
    },
)