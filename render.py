#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import utils.vis_utils as VISUils

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    ### Optional: save depth/normal maps
    # render, mask, expected_coord, median_coord, expected_depth, median_depth, 
    # viewspace_points, viewspace_points, radii, normal
    render_outputs = ['mask', 'expected_depth', 'median_depth', 'normal']
    outputs_path = []
    for output_idx in render_outputs:
        output_idx_path = os.path.join(model_path, name, "ours_{}".format(iteration), f'renders_{output_idx}')
        makedirs(output_idx_path, exist_ok=True)
        outputs_path.append(output_idx_path)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_dict = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
        rendering = render_dict["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        ### Optional: save depth/normal maps
        for jdx, output_jdx in enumerate(render_outputs):
            render_output = render_dict[output_jdx]
            
            if 'mask' in output_jdx:
                torchvision.utils.save_image(render_output, os.path.join(outputs_path[jdx], '{0:05d}'.format(idx) + ".png"))
            elif 'depth' in output_jdx:
                render_output_map = VISUils.apply_depth_colormap(render_output[0,...,None], render_dict['mask'][0,...,None], near_plane=0.0, far_plane=5.0).detach()
                torchvision.utils.save_image(render_output_map.permute(2,0,1), os.path.join(outputs_path[jdx], '{0:05d}'.format(idx) + ".png"))
            elif 'normal' in output_jdx:
                # transform normal from view space to world space
                render_output = (render_output.permute(1,2,0) @ (view.world_view_transform[:3,:3].T)).permute(2,0,1)

                render_output_map = ((render_output+1)/2).clip(0, 1)
                torchvision.utils.save_image(render_output_map, os.path.join(outputs_path[jdx], '{0:05d}'.format(idx) + ".png"))
            else:
                pass

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.kernel_size)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.kernel_size)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)