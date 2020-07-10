apt-get update && \
    apt-get install -y --no-install-recommends \
    curl ca-certificates bzip2 procps  

curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p/opt/conda && \
    rm ~/miniconda.sh 
echo "PATH=/opt/conda/bin:$PATH" >> /root/.bashrc

apt-get install -y --no-install-recommends \
    neovim gdb wget man-db tree silversearcher-ag build-essential \
    git ssh-client 

git config --global user.name "Andrew Jones" && \
    git config --global user.email "andyjones.ed@gmail.com"

pip install \
    numpy torch torchvision tqdm matplotlib beautifulsoup4 shapely rasterio psutil ninja \
    pandas jupyter seaborn bokeh setproctitle wurlitzer ipython==7.5 av flake8 rope gitpython \
    sphinx sphinx-rtd-theme

# Copy the Jupyter config into place. 
cp -rf docker/.jupyter /root/.jupyter
cp -rf docker/.ipython /root/.ipython

pip install git+https://github.com/andyljones/aljpy.git && \
    pip install git+https://github.com/andyljones/snakeviz@custom && \
    pip install git+https://github.com/andyljones/noterminal && \
    pip install git+https://github.com/andyljones/pytorch_memlab && \
    rm -rf ~/.cache

# Install my frontend Jupyter extensions 
pip install jupyter_contrib_nbextensions && \
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