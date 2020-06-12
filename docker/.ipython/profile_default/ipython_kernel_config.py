## List of files to run at IPython startup.
c.IPKernelApp.exec_files = []

## lines of code to run at IPython startup.
c.IPKernelApp.exec_lines = [r'%load_ext autoreload', r'%autoreload 2', r'%load_ext snakeviz']

c.IPKernelApp.matplotlib = "inline"
c.InlineBackend.rc = {'figure.figsize': (12, 8)}
