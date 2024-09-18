# The evalution code is fork from GOF https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/evaluate_dtu_mesh.py
import numpy as np
import torch
import torch.nn.functional as F
from scene import Scene
import cv2
import os
import random
import glob
from os import makedirs, path
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import trimesh
from skimage.morphology import binary_dilation, disk

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def load_dtu_camera(DTU):
    # Load projection matrix from file.
    camtoworlds = []
    for i in range(1, 64+1):
        fname = path.join(DTU, f'Calibration/cal18/pos_{i:03d}.txt')

        projection = np.loadtxt(fname, dtype=np.float32)

        # Decompose projection matrix into pose and camera matrix.
        camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
        camera_mat = camera_mat / camera_mat[2, 2]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_mat.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        pose = pose[:3]
        camtoworlds.append(pose)
    return camtoworlds

import math
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def cull_mesh_radegs(cameras, mesh):
    '''
        origin code of culling the mesh from rade-gs
    '''
    
    vertices = mesh.vertices
    
    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()
    
    sampled_masks = []
    
    for camera in cameras:
        c2w = (camera.world_view_transform.T).inverse()
        w2c = torch.inverse(c2w).cuda()
        mask = camera.gt_mask
        fx = fov2focal(camera.FoVx, camera.image_width)
        fy = fov2focal(camera.FoVy, camera.image_height)
        intrinsic = torch.eye(4)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 2] = camera.image_width / 2.
        intrinsic[1, 2] = camera.image_height / 2.
        intrinsic = intrinsic.cuda()

        W, H = camera.image_width, camera.image_height
        
        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()
            
            # dialate mask similar to unisurf
            maski = mask[0, :, :].cpu().numpy().astype(np.float32) / 256.
            # maski = torch.from_numpy(binary_dilation(maski, disk(14))).float()[None, None].cuda()
            maski = torch.from_numpy(binary_dilation(maski, disk(6))).float()[None, None].cuda()
            
            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

            sampled_mask = sampled_mask + (1. - valid)

            sampled_masks.append(sampled_mask)
        
    sampled_masks = torch.stack(sampled_masks, -1)

    # filter
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)
    
    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    return mesh

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def cull_scan(scan, mesh_path, result_mesh_file, instance_dir):
    '''
        code of culling the mesh from 2dgs
    '''
    
    # load poses
    image_dir = '{0}/images'.format(instance_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    n_images = len(image_paths)
    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())
    
    # load mask
    mask_dir = '{0}/mask'.format(instance_dir)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p)
        masks.append(mask)

    # hard-coded image shape
    W, H = 1600, 1200

    # load mesh
    mesh = trimesh.load(mesh_path)
    mesh_origin = mesh.copy()
    
    # load transformation matrix

    vertices = mesh.vertices

    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()

    sampled_masks = []
    for i in tqdm(range(n_images),  desc="Culling mesh given masks"):
        pose = pose_all[i]
        w2c = torch.inverse(pose).cuda()
        intrinsic = intrinsics_all[i].cuda()

        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()
            
            # dialate mask similar to unisurf
            maski = masks[i][:, :, 0].astype(np.float32) / 256.
            maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()

            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

            sampled_mask = sampled_mask + (1. - valid)
            sampled_masks.append(sampled_mask)

    sampled_masks = torch.stack(sampled_masks, -1)
    # filter
    
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)

    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    
    # transform vertices to world 
    scale_mat = scale_mats[0]
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    print(result_mesh_file)
    mesh.export(result_mesh_file)
    del mesh

    # transfrom mesh_origin to world
    mesh_origin.vertices = mesh_origin.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    mesh_origin.export(result_mesh_file.split('culled.ply')[0] + 'aligned.ply')
    del mesh_origin

def evaluate_mesh(dataset : ModelParams, iteration : int, DTU_PATH : str):
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    dtu_cameras = load_dtu_camera(args.DTU)
    
    gt_points = np.array([cam[:, 3] for cam in dtu_cameras])
    
    points = []
    for cam in train_cameras:
        c2w = (cam.world_view_transform.T).inverse()
        points.append(c2w[:3, 3].cpu().numpy())
    points = np.array(points)
    gt_points = gt_points[:points.shape[0]]
    
    # align the scale of two point clouds
    scale_points = np.linalg.norm(points - points.mean(axis=0), axis=1).mean()
    scale_gt_points = np.linalg.norm(gt_points - gt_points.mean(axis=0), axis=1).mean()
    
    points = points * scale_gt_points / scale_points
    _, r, t = best_fit_transform(points, gt_points)
    
    scan = int(dataset.source_path.split("/")[-1][4:])

    # load mesh
    # mesh_file = os.path.join(dataset.model_path, "test/ours_{}".format(iteration), mesh_dir, filename)
    # todo TSDF fusion w/o mask
    mesh_file = os.path.join(dataset.model_path, "recon_womask_post.ply")

    print("cull mesh ....")
    result_mesh_file = os.path.join(dataset.model_path, "recon_womask_post_culled.ply")
    cull_scan(scan, mesh_file, result_mesh_file, instance_dir=dataset.source_path)

    # align the mesh, rade-gs
    # print("load")
    # mesh = trimesh.load(mesh_file)

    # print("cull")
    # mesh = cull_scan(train_cameras, mesh)

    # culled_mesh_file = os.path.join(dataset.model_path, "recon_womask_culled.ply")
    # mesh.export(culled_mesh_file)
    
    # mesh.vertices = mesh.vertices * scale_gt_points / scale_points
    # mesh.vertices = mesh.vertices @ r.T + t
    
    # aligned_mesh_file = os.path.join(dataset.model_path, "recon_womask_aligned.ply")
    # mesh.export(aligned_mesh_file)
        
    # evaluate
    out_dir = os.path.join(dataset.model_path, "vis")
    os.makedirs(out_dir,exist_ok=True)
    # scan = dataset.model_path.split("/")[-1][4:]
    
    cmd = f"python dtu_eval/eval.py --data {result_mesh_file} --scan {scan} --mode mesh --dataset_dir {DTU_PATH} --vis_out_dir {out_dir}"
    print(cmd)
    os.system(cmd)

    origin_mesh_file = result_mesh_file.split('culled.ply')[0] + 'aligned.ply'
    cmd = f"python dtu_eval/eval.py --data {origin_mesh_file} --scan {scan} --mode mesh --dataset_dir {DTU_PATH} --vis_out_dir {out_dir} --suffix_name womask"
    print(cmd)
    os.system(cmd) 
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30_000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--scan_id', type=str,  help='scan id of the input mesh')
    parser.add_argument('--DTU', type=str,  default='dtu_eval/Offical_DTU_Dataset', help='path to the GT DTU point clouds')
    
    args = get_combined_args(parser)
    print("evaluating " + args.model_path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    evaluate_mesh(model.extract(args), args.iteration, args.DTU)