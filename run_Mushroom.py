import os

# Mushroom dataset
mushroom_root = '../../Data/MuSHRoom_iphone-loftr'
mushroom_exp = '../exps/full/MuSHRoom_iphone-loftr'
mushroom_scenes = ['koivu', 'honka', 'computer']
cuda_device = 0

# train
cmd_lis=[]
for scene in mushroom_scenes:
    # cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {mushroom_root}/{scene} -m {mushroom_exp}/{scene} -r 2 --eval')
    # cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python render.py -s {mushroom_root}/{scene} -m {mushroom_exp}/{scene} -r 2 --eval')

    # cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python metric.py -m {mushroom_exp}/{scene} -f train')
    # cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python metric.py -m {mushroom_exp}/{scene} -f test')
    # cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python vis_outputs.py -m {mushroom_exp}/{scene} -f train test')

    # cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python mesh_extract.py -s {mushroom_root}/{scene} -m {mushroom_exp}/{scene} -r 2 --depth_trunc 5.0 --voxel_size 0.01')
    cmd_lis.append(f'CUDA_VISIBLE_DEVICES={cuda_device} python mesh_extract.py -s {mushroom_root}/{scene} -m {mushroom_exp}/{scene} -r 2 --mesh_type poisson --poisson_depth 10.0')

# run cmd
for cmd in cmd_lis:
    print(cmd)
    os.system(cmd)
