Build docs: `sphinx-build -b html docs docs/_build`
Serve docs: `docker exec -it onedee python -m http.server --directory /code/docs/_build 9095`