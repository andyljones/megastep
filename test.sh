#!/bin/bash

# Build the package
python setup.py clean --all sdist bdist_wheel

# Install the package
pip uninstall megastep -y
pip install $(ls dist/*.whl)[cubicasa,rebar,test]

# Drop into a dir where the codebase isn't visible
mkdir -p test
cd test

path=$(python -c "import megastep; print(megastep.__file__[:-12])")
pytest -o python_files=*.py $path