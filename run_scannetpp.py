import os

# Mushroom dataset
scannetpp_root = '../../Data/ScanNetpp'
scannetpp_exp = '../exps/full/scannetpp'
# scannetpp_scenes = ['8b5caf3398', '116456116b', '13c3e046d7', '0a184cf634', '578511c8a9', '21d970d8de']
scannetpp_scenes = ['8b5caf3398']
cuda_device = 0

# train
cmd_lis=[]
for scene in scannetpp_scenes:
    cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {scannetpp_root}/{scene} -m {scannetpp_exp}/{scene} -r 2 --eval')
    cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python render.py -s {scannetpp_root}/{scene} -m {scannetpp_exp}/{scene} -r 2 --eval')

    # NVS metrics and visualization 
    cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python metric.py -m {scannetpp_exp}/{scene} -f train')
    cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python metric.py -m {scannetpp_exp}/{scene} -f test')
    cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python vis_outputs.py -m {scannetpp_exp}/{scene} -f train test')

    # TSDF Fusion
    cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python mesh_extract.py -s {scannetpp_root}/{scene} -m {scannetpp_exp}/{scene} -r 2 --depth_trunc 5.0 --voxel_size 0.01')
    # Poisson Reconstruction
    cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python mesh_extract.py -s {scannetpp_root}/{scene} -m {scannetpp_exp}/{scene} -r 2 --mesh_type poisson --poisson_depth 8.0')
    # Marching Tetrahedra
    cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python mesh_extract_tetrahedra.py -s {scannetpp_root}/{scene} -m {scannetpp_exp}/{scene} -r 2 --eval')

    # evaluate mesh, depth & normal
    # common_args = " -f train test -p tsdf poisson tetrahedra"
    # cmd_lis.append(f"CUDA_VISIBLE_DEVICES={cuda_device} python eval_geometry.py -s {scannetpp_root}/{scene} -m {scannetpp_exp}/{scene}" + common_args)

# run cmd
for cmd in cmd_lis:
    print(cmd)
    os.system(cmd)
