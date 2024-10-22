#!/usr/bin/env python

from distutils.core import setup

setup(name='stratified_estimator',
  version='0.1',
  description='Stratified space estimator tools',
  author='Michael Robinson',
  author_email='kb1dds@gmail.com',
  url='https://github.com/kb1dds/stratified_estimator',
  packages=['stratified_estimator'],
  install_requires=['numpy','scipy','torch']
 )
