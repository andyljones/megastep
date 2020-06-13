# Can't upgrade to 10.2 until PyTorch supports it (1.5?)
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# Grab tini so that Jupyter doesn't spray zombies everywhere
ADD https://github.com/krallin/tini/releases/download/v0.18.0/tini /usr/bin/tini
# Get the rsyncd config
RUN apt-get update && \
    # There's something weird going on with this particular image and fetching this particular library
    # Installing it separately first works; installing it implicitly with all the other deps below breaks
    # with a 400 Bad Request error. Might be transient.
    apt-get install -y --no-install-recommends \
        # Needed for sanity
        neovim gdb wget man-db tree silversearcher-ag build-essential \
        # Needed for conda
        curl ca-certificates bzip2 procps \ 
        # Needed for git installs
        git ssh-client \
        # Needed for fast PyTorch linking
        lld ccache && \
    # Set up tini
    chmod +x /usr/bin/tini

# Softlink LLD into place. Apparently makes a difference with large CUDA projects
# https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
RUN ln -sf /usr/lib/llvm-6.0/bin/ld.lld /usr/bin/ld && \
    /usr/sbin/update-ccache-symlinks && \
    ln -s /usr/bin/ccache /usr/lib/ccache/nvcc && \
    ccache -M 5Gi && \
    ccache -F 0
ENV PATH $PATH:/usr/lib/ccache

# Set up git
RUN git config --global user.name "Andrew Jones" && \
    git config --global user.email "andyjones.ed@gmail.com"

# Set up conda
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Copy the Jupyter config into place. 
ADD .jupyter /root/.jupyter
ADD .ipython /root/.ipython

# Install the Python packages I usually use
RUN conda install -y numpy pytorch torchvision pandas -c pytorch && \
    conda clean -ya && \
    # Install Jupyter 7.5 because 7.6.1 has a bunch of lag with autoreload 
    pip install scipy psutil jupyter tqdm seaborn bokeh matplotlib setproctitle wurlitzer ipython==7.5 av flake8 \
        beautifulsoup4 shapely rasterio gym

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

WORKDIR /code

ENTRYPOINT ["tini", "--", "run.sh"]