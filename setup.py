from setuptools import find_packages, setup

setup(
    name='ddm_dynamical',
    packages=find_packages(
        include=["ddm_dynamical"]
    ),
    version='0.1.0',
    description='Package for denoising diffusion models of dynamical systems.',
    author='Tobias Finn',
    license='MIT',
)
