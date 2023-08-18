#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup

PACKAGE_NAME = "idr_accelerate"
VERSIONFILE = "VERSION"
AUTHOR = "IDRIS"
AUTHOR_EMAIL = "assist@idris.fr"
URL = "https://www.idris.fr"

with open(os.path.join(PACKAGE_NAME, VERSIONFILE), "r") as file:
    VERSION = file.read().strip()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=find_packages(),
    entry_points={
        "console_scripts": ["idr_accelerate = idr_accelerate.launcher:run"],
    },
    install_requires=[
        "accelerate",
        "idr_torch>=2.0.0",
        "packaging",
        "torch",
    ],
    include_package_data=True,
    license="MIT",
)
