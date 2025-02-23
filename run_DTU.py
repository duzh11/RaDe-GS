import os

# DTU
# scan24, scan37, scan40, scan55, scan63, scan65, scan69, scan83, 
# scan97, scan105, scan106, scan110, scan114, scan118, scan122

dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', \
              'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']

# train
for scene in dtu_scenes:
    cmd_lis=[]
    # cmd_lis.append(f'python train.py -s ../../Data/DTU/{scene} -m ../exps/full/DTU/{scene} -r 2')
    cmd_lis.append(f'python mesh_extract.py -s ../../Data/DTU/{scene} -m ../exps/full/DTU/{scene} -r 2 --voxel_size 0.002 --sdf_trunc 0.016 --depth_trunc 3.0 --usingmask')
    cmd_lis.append(f'python evaluate_dtu_mesh.py --DTU ../../Data/Offical_DTU_Dataset -s ../../Data/DTU/{scene} -m ../exps/full/DTU/{scene}')
    
    cmd_lis.append(f'python render.py -s ../../Data/DTU/{scene} -m ../exps/full/DTU/{scene}')
    # cmd_lis.append(f'python metric.py -m ../exps/full/DTU/{scene} -f train')
    cmd_lis.append(f'python vis_outputs.py -f train -m ../exps/full/DTU/{scene}')
    
    # run cmd
    for cmd in cmd_lis:
        os.system(cmd)
