from setuptools import setup, find_packages

setup(
    name="janc",
    version="0.1",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'cantera',
        'amr @ git+https://github.com/luofx23/JAX-AMR.git@main#subdirectory=src/amr'
    ],
    python_requires=">=3.7"
)