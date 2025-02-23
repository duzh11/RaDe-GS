import os

# Mushroom dataset
TnTadvanced_root = '../../Data/TanksAndTemples_advanced-loftr'
TnTadvanced_exp = '../exps/full/TanksAndTemples_advanced-loftr'
TnTadvanced_scenes = ['Auditorium', 'Courtroom', 'Ballroom']
cuda_device = 0

# train
cmd_lis=[]
for scene in TnTadvanced_scenes:
    # cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {TnTadvanced_root}/{scene} -m {TnTadvanced_exp}/{scene} -r 2 --eval')
    # cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python render.py -s {TnTadvanced_root}/{scene} -m {TnTadvanced_exp}/{scene} -r 2 --eval')

    # cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python metric.py -m {TnTadvanced_exp}/{scene} -f train')
    # cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python metric.py -m {TnTadvanced_exp}/{scene} -f test')
    # cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python vis_outputs.py -m {TnTadvanced_exp}/{scene} -f train test')

    # cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python mesh_extract.py -s {TnTadvanced_root}/{scene} -m {TnTadvanced_exp}/{scene} -r 2 --depth_trunc 20.0 --voxel_size 0.01')
    cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python mesh_extract.py -s {TnTadvanced_root}/{scene} -m {TnTadvanced_exp}/{scene} -r 2 --mesh_type poisson --poisson_depth 15.0')

# run cmd
for cmd in cmd_lis:
    print(cmd)
    os.system(cmd)
