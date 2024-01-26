from setuptools import setup

setup(
    name='ctao_cosmic_ray_spectra',
    version='0.1',
    packages=['ctao_cosmic_ray_spectra'],
    install_requires=[
        "numpy",
        "astropy",
        "importlib",
        "scipy",
        "pytest"
        'setuptools>=40.0',
    ],
)