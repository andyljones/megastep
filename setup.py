#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name='megastep',
    version='0.2.3',
    description='Helps build million-frame-per-second reinforcement learning environments',
    author='Andy Jones',
    author_email='andyjones.ed@gmail.com',
    url='http://andyljones.com/megastep',
    packages=find_packages(include=['megastep*', 'rebar*']),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.18',
        'torch>=1.5',
        'torchvision>=0.6',
        'tqdm>=4',
        'matplotlib>=3',
        'ninja>=1.10'],
    extras_require={
        'cubicasa': [
            'beautifulsoup4>=4', 
            'shapely>=1.7', 
            'rasterio>=1.1',
            'pandas>=1'],
        'rebar': [
            'av>=8',
            'bokeh>=2',
            'ipywidgets>=7',
            'psutil>=5',
            'pandas>=1'],
        'docs': [
            'sphinx>=3'],
        'test': [
            'pytest>=5']},
    package_data={'megastep': ['src/*']}
)