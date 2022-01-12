#!/usr/bin/python3

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build platform specific wheel files for the iree.runtime package.
# Built artifacts are per-platform and build out of the build tree.

from distutils.command.install import install
import os
import platform
from setuptools import setup, find_namespace_packages

README = r'''
TensorFlow TFLite Compiler Tools
'''

exe_suffix = ".exe" if platform.system() == "Windows" else ""
import_tflite_path = os.path.join(os.path.dirname(__file__), "iree", "tools",
                                  "tflite", f"iree-import-tflite{exe_suffix}")
if not os.access(import_tflite_path, os.X_OK):
  raise RuntimeError(
      f"Tool not found ({import_tflite_path}). Be sure to build "
      f"//iree_tf_compiler:iree-import-tflite and run ./symlink_binaries.sh")

# Force platform specific wheel.
# https://stackoverflow.com/questions/45150304
try:
  from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

  class bdist_wheel(_bdist_wheel):

    def finalize_options(self):
      _bdist_wheel.finalize_options(self)
      self.root_is_pure = False

    def get_tag(self):
      python, abi, plat = _bdist_wheel.get_tag(self)
      # We don't contain any python extensions so are version agnostic
      # but still want to be platform specific.
      python, abi = 'py3', 'none'
      return python, abi, plat

except ImportError:
  bdist_wheel = None


# Force installation into platlib.
# Since this is a pure-python library with platform binaries, it is
# mis-detected as "pure", which fails audit. Usually, the presence of an
# extension triggers non-pure install. We force it here.
class platlib_install(install):

  def finalize_options(self):
    install.finalize_options(self)
    self.install_lib = self.install_platlib


setup(
    name="iree-tools-tflite",
    version="0.1",
    author="The IREE Team",
    author_email="iree-discuss@googlegroups.com",
    license="Apache",
    description="IREE TFLite Compiler Tools",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/google/iree",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.7",
    packages=find_namespace_packages(include=[
        "iree.tools.tflite",
        "iree.tools.tflite.*",
    ]),
    package_data={
        "iree.tools.tflite": [f"iree-import-tflite{exe_suffix}",],
    },
    cmdclass={
        'bdist_wheel': bdist_wheel,
        'install': platlib_install,
    },
    entry_points={
        "console_scripts": [
            "iree-import-tflite = iree.tools.tflite.scripts.iree_import_tflite.__main__:main",
        ],
    },
    zip_safe=False,  # This package is fine but not zipping is more versatile.
)