#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name='megastep',
    version='0.1',
    description='Helps build million-frame-per-second reinforcement learning environments',
    author='Andy Jones',
    author_email='andyjones.ed@gmail.com',
    url='www.github.com/andyljones/megastep',
    packages=find_packages(include=['megastep.*', 'rebar.*', 'cubicasa.*', 'demo.*']),
    install_requires=[
        'numpy>=1.18',
        'torch>=1.5',
        'torchvision>=0.6',
        'tqdm>=4',
        'matplotlib>=3'],
    extras_require={
        'cubicasa': [
            'beautifulsoup4>=4', 
            'shapely>=1.7', 
            'rasterio>=1.1',
            'pandas>=1'],
        'rebar': [
            'av>=8',
            'bokeh>=2'],
        'docs': [
            'sphinx>=3'
        ]}
      )