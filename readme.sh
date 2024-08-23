# ================================= # 
# ============ install ============ # 
# ================================= # 

pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/

# tetra-nerf for Marching Tetrahedra
cd submodules/tetra_triangulation
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal
mkdir build
cd build
cmake ..
# you can specify your own cuda path
export CPATH=/usr/local/cuda-11.3/targets/x86_64-linux/include:$CPATH

make 
pip install -e .

# ======================================== # 
# ============ train and eval ============ # 
# ======================================== # 

# +++++++++ DTU +++++++++ # 
python train.py -s ../../Data/DTU/scan24 -m ../exps/DTU/scan24 -r 2 --use_decoupled_appearance
python mesh_extract.py -s ../../Data/DTU/scan24 -m ../exps/DTU/scan24 -r 2
python evaluate_dtu_mesh.py -s ../../Data/DTU/scan24 -m ../exps/DTU/scan24

# +++++++++ TNT +++++++++ # 
python train.py -s ../../Data/TNT/Barn -m ../exps/TNT/Barn -r 2 --eval --use_decoupled_appearance
python mesh_extract_tetrahedra.py -s ../../Data/TNT/Barn -m ../exps/TNT/Barn -r 2 --eval