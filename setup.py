#!/usr/bin/env python

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='pepg-es',
      version='0.0.5',
      description='Python Implementation of Parameter-exploring Policy Gradients Evolution Strategy',
      author='Göktuğ Karakaşlı',
      author_email='karakasligk@gmail.com',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/goktug97/PEPG-ES',
      download_url=(
          'https://github.com/goktug97/PEPG-ES/archive/v0.0.5.tar.gz'),
      packages = ['pepg'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: POSIX :: Linux",
      ],
      install_requires=[
          'numpy'
      ],
      python_requires='>=3.6',
      include_package_data=True)
