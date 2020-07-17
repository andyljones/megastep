#!/bin/sh

# Copied from https://stackoverflow.com/questions/48588908/deploying-ignored-dist-folder-to-github-pages

sphinx-build -E -b html docs docs/_build
touch docs/_build/.nojekyll
git branch --delete --force gh-pages
git checkout --orphan gh-pages
git add -f docs/_build
git commit -m "Rebuild GitHub pages"
git filter-branch -f --prune-empty --subdirectory-filter docs/_build && git push -f origin gh-pages && git checkout master