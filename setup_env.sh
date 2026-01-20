conda create -n ngff python=3.10 -y
conda activate ngff
pip install -r requirements.txt
git clone https://github.com/BaowenZ/RaDe-GS.git --recursive && cd RaDe-GS/submodules
pip install ./diff-gaussian-rasterization
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu126.html