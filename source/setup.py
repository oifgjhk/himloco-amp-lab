# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'himloco_lab' python package."""

import os
import toml

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
#
# NOTE: The Isaac Lab ecosystem packages (isaaclab, isaaclab_assets, isaaclab_rl,
# isaaclab_tasks, isaaclab_mimic) are managed through the Isaac Lab extension system
# (extension.toml) and must be installed BEFORE this package. They are NOT listed here.
#
# NOTE: torch, torchvision, and other Isaac Sim bundled packages are provided by the
# Isaac Sim Python environment. They are listed here with minimum version constraints
# for documentation and for users running outside of Isaac Sim (e.g. for testing).
INSTALL_REQUIRES = [
    # Core numerical & scientific
    "numpy>=1.22",
    "scipy>=1.10",
    # ML framework (bundled with Isaac Sim; listed for non-Isaac-Sim installs)
    "torch>=2.0",
    # Utilities
    "psutil==5.9.8",
    "tqdm>=4.60",
    "pyyaml>=6.0",
    "toml>=0.10",
    "packaging>=21.0",
    "protobuf>=3.20",
    # Logging & visualization
    "tensorboard>=2.10",
    "matplotlib>=3.5",
]

# Installation operation
setup(
    name="himloco_lab",
    packages=find_packages(include=["himloco_lab", "himloco_lab.*"]),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)