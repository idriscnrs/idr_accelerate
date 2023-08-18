#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="idr_accelerate",
    version="2.0.0",
    author="IDRIS",
    author_email="assist@idris.fr",
    url="https://www.idris.fr",
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
    license="MIT",
)
