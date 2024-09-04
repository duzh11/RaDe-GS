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

# DTU
# scan24, scan37, scan40, scan55, scan63, scan65, scan69, scan83, 
# scan97, scan105, scan106, scan110, scan114, scan118, scan122

# TNT
# Barn, Caterpillar, Courthouse, Ignatius, Meetingroom, Truck

# +++++++++ DTU +++++++++ # 
python train.py -s ../../Data/DTU/scan24 -m ../exps/full/DTU/scan24 -r 2 --use_decoupled_appearance
python mesh_extract.py -s ../../Data/DTU/scan24 -m ../exps/full/DTU/scan24 -r 2
python evaluate_dtu_mesh.py --DTU ../../Data/Offical_DTU_Dataset -s ../../Data/DTU/scan24 -m ../exps/full/DTU/scan24

python render.py -s ../../Data/DTU/scan24 -m ../exps/full/DTU/scan24 
python metric.py -m ../exps/full/DTU/scan24 -f train

# +++++++++ TNT +++++++++ # 
python train.py -s ../../Data/TNT/Barn -m ../exps/full/TNT/Barn -r 2 --eval --use_decoupled_appearance
python mesh_extract_tetrahedra.py -s ../../Data/TNT/Barn -m ../exps/full/TNT/Barn -r 2 --eval
python eval_tnt/run.py --dataset-dir ../../Data/TNT_dataset/Barn --traj-path ../../Data/TNT_dataset/Barn/Barn_COLMAP_SfM.log --ply-path ../exps/full/TNT/Barn/recon_tetrahedra.ply

python render.py -s ../../Data/TNT/Barn -m ../exps/full/TNT/Barn -r 2 --eval
python metric.py -m ../exps/full/TNT/Barn -f train
python metric.py -m ../exps/full/TNT/Barn -f test

# ============ concat and vis ============ # 
python vis_outputs.py -f 'train' -m ../exps/full/DTU/scan24 ../exps/full/DTU/scan37
