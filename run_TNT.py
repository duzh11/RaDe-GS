import os

# TNT
# Barn, Caterpillar, Courthouse, Ignatius, Meetingroom, Truck

tnt_scenes = ['Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck']

# train
for scene in tnt_scenes:
    cmd_lis=[]
    # cmd_lis.append(f'python train.py -s ../../Data/TNT/{scene} -m ../exps/full/TNT/{scene} -r 2 --eval')
    # cmd_lis.append(f'python mesh_extract_tetrahedra.py -s ../../Data/TNT/{scene} -m ../exps/full/TNT/{scene} -r 2 --eval')
    cmd_lis.append(f'python eval_tnt/run.py --dataset-dir \
        ../../Data/Official_TNT_dataset/{scene} \
        --traj-path ../../Data/Official_TNT_dataset/{scene}/{scene}_COLMAP_SfM.log \
        --ply-path ../exps/full/TNT/{scene}/recon_tetrahedra.ply')


    # cmd_lis.append(f'python render.py -s ../../Data/TNT/{scene} -m ../exps/full/TNT/{scene} -r 2 --eval')
    # cmd_lis.append(f'python metric.py -m ../exps/full/TNT/{scene} -f train')
    # cmd_lis.append(f'python metric.py -m ../exps/full/TNT/{scene} -f test')
    # cmd_lis.append(f'python vis_outputs.py -f train test -m ../exps/full/TNT/{scene}')
    
    # run cmd
    for cmd in cmd_lis:
        os.system(cmd)
