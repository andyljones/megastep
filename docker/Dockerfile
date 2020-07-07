# Build image
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 as build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Needed for conda
        curl ca-certificates bzip2 procps \ 

# Set up conda
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install the Python packages I usually use
RUN conda install -y numpy pytorch torchvision -c pytorch
RUN pip install 
        # Core reqs
        tqdm matplotlib beautifulsoup4 shapely rasterio \ 
        # Docs
        sphinx sphinx-rtd-theme

WORKDIR /code


# Dev image
FROM build AS dev

RUN apt-get install -y --no-install-recommends \
        # Needed for sanity
        neovim gdb wget man-db tree silversearcher-ag build-essential \
        # Needed for git installs
        git ssh-client 

# Grab tini so that Jupyter doesn't spray zombies everywhere
ADD https://github.com/krallin/tini/releases/download/v0.18.0/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Set up git
RUN git config --global user.name "Andrew Jones" && \
    git config --global user.email "andyjones.ed@gmail.com"

# Install the Python packages I usually use
# Install Jupyter 7.5 because 7.6.1 has a bunch of lag with autoreload 
RUN pip install pandas jupyter tqdm seaborn bokeh matplotlib setproctitle wurlitzer ipython==7.5 av flake8 \
        beautifulsoup4 shapely rasterio gym rope gitpython \ 
        # Docs
        sphinx sphinx-rtd-theme

# Copy the Jupyter config into place. 
ADD .jupyter /root/.jupyter
ADD .ipython /root/.ipython

# Install my backend Jupyter extensions
# aljpy needs to be before noterminal
RUN pip install git+https://github.com/andyljones/aljpy.git && \ 
    pip install git+https://github.com/andyljones/snakeviz@custom && \
    pip install git+https://github.com/andyljones/noterminal && \
    pip install git+https://github.com/andyljones/pytorch_memlab && \
    rm -rf ~/.cache

# Install my frontend Jupyter extensions 
RUN pip install jupyter_contrib_nbextensions && \ 
    jupyter contrib nbextension install --user && \
    cd /root && mkdir nbextensions && cd nbextensions && \
    git clone https://github.com/andyljones/nosearch && \
    cd nosearch && \
    jupyter nbextension install nosearch && \
    jupyter nbextension enable nosearch/main && \
    cd .. && \
    git clone https://github.com/andyljones/noterminal && \
    cd noterminal && \
    jupyter nbextension install noterminal && \
    jupyter nbextension enable noterminal/main && \
    cd .. && \
    git clone https://github.com/andyljones/stripcommon && \
    cd stripcommon && \
    jupyter nbextension install stripcommon && \
    jupyter nbextension enable stripcommon/main && \
    jupyter nbextension enable autoscroll/main 

# Set up the entrypoint script
ADD run.sh /usr/bin/

ENTRYPOINT ["tini", "--", "run.sh"]
