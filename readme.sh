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
python train.py -s ../../Data/DTU/scan24 -m ../exps/full/DTU/scan24 -r 2 --use_decoupled_appearance
python mesh_extract.py -s ../../Data/DTU/scan24 -m ../exps/full/DTU/scan24 -r 2
python evaluate_dtu_mesh.py -s ../../Data/DTU/scan24 -m ../exps/full/DTU/scan24

python render.py -s ../../Data/DTU/scan24 -m ../exps/full/DTU/scan24 
python metric.py -m ../exps/full/DTU/scan24 -f train

# +++++++++ TNT +++++++++ # 
python train.py -s ../../Data/TNT/Barn -m ../exps/full/TNT/Barn -r 2 --eval --use_decoupled_appearance
python mesh_extract_tetrahedra.py -s ../../Data/TNT/Barn -m ../exps/full/TNT/Barn -r 2 --eval
# python eval_tnt/run.py --dataset-dir <path to GT TNT dataset> --traj-path <path to preprocessed TNT COLMAP_SfM.log file> --ply-path <output folder>/recon.ply

python render.py -s ../../Data/TNT/Barn -m ../exps/full/TNT/Barn -r 2 --eval
python metric.py -m ../exps/full/TNT/Barn  -f train
python metric.py -m ../exps/full/TNT/Barn  -f test