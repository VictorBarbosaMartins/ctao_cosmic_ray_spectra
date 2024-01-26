from setuptools import setup, find_packages
import os


extras_require = {

    "tests": [
        "pytest",
        "pytest-cov",
        "astropy",
    ],
}
extras_require["dev"] = extras_require["tests"] + [
    "setuptools_scm",
]

all_extras = set()
for extra in extras_require.values():
    all_extras.update(extra)
extras_require["all"] = list(all_extras)

setup(
    use_scm_version={"write_to": os.path.join("ctao_cosmic_ray_spectra", "_version.py")},
    packages=find_packages(exclude=['ctao_cosmic_ray_spectra._dev_version']),
    install_requires=[
        "astropy>=5.3,<7.0.0a0",
        "numpy>=1.21",
        "scipy<1.12",
    ],
    include_package_data=True,
    extras_require=extras_require,
)