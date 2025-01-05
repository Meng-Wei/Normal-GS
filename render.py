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
import os
import torch

import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import linear_to_srgb
from utils.graphics_utils import normalize_rendered_by_weights, render_normal_from_depth
from utils.loss_utils import normal_diff

Measure_Normal = True

def render_set_normal(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    full_dict = {}
    normal_consistency = []
    full_dict["Normal Consistency"] = 0.0
    full_dict["Depth Diff"] = 0.0
    full_dict["Weighted Depth Diff"] = 0.0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, render_n=True, render_full=True)
        torch.cuda.synchronize(); t1 = time.time()

        normal = render_pkg["normal"]
        alpha = render_pkg["alpha"]
        depth = render_pkg["depth"]

        alpha = alpha[0]
        depth = depth[0]

        surface_mask = alpha > 0.5

        unweighted_normal = normalize_rendered_by_weights(normal, alpha, 0.5)
        torchvision.utils.save_image((unweighted_normal + 1.0 ) / 2.0, os.path.join(render_path, '{0:05d}_unw'.format(idx) + ".png"))
        # torchvision.utils.save_image((unweighted_normal + 1.0 ) * surface_mask / 2.0, os.path.join(render_path, '{0:05d}_unw'.format(idx) + ".png"))

        final_normal = torch.nn.functional.normalize(unweighted_normal, dim=0)
        torchvision.utils.save_image((final_normal + 1.0 ) / 2.0, os.path.join(render_path, '{0:05d}_final'.format(idx) + ".png"))
        # torchvision.utils.save_image((final_normal + 1.0 ) * surface_mask / 2.0, os.path.join(render_path, '{0:05d}_final'.format(idx) + ".png"))

        normal_from_depth = render_normal_from_depth(view, depth)
        torchvision.utils.save_image((normal_from_depth + 1.0 ) / 2.0, os.path.join(render_path, '{0:05d}_depth_normal'.format(idx) + ".png"))

        depth = depth / (depth.max() + 1e-5)
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

        if Measure_Normal and view.normal is not None:
            if view.normal is not None:
                normal_ref = torch.from_numpy(view.normal).cuda().permute(2, 0, 1) * 2.0 - 1.0
                mvs_mask = torch.zeros_like(normal_ref)
                mvs_mask[:, torch.norm(normal_ref, dim=0) > 0.9] = 1
                mvs_mask = mvs_mask[0]
                mvs_mask[surface_mask < 1] = 0

                # if idx == 0:
                #     print("    normal,", normal_diff(normal, normal_ref, mvs_mask))
                #     print("    unweighted_normal,", normal_diff(unweighted_normal, normal_ref * 2.0 - 1.0, mvs_mask))
                #     print("    final_normal,", normal_diff(final_normal, normal_ref * 2.0 - 1.0, mvs_mask))
                #     print("    normal_from_depth,", normal_diff(normal_from_depth, normal_ref * 2.0 - 1.0, mvs_mask))
                #     print("\n\n")

                dotAngle = normal_diff(final_normal, normal_ref, mvs_mask)
                normal_consistency.append(dotAngle)

    if Measure_Normal and len(views) > 0 and view.normal is not None:
        print("  Normal Consistency : {:>12.7f}".format(torch.tensor(normal_consistency).mean(), ".5"))

        full_dict.update({
            "Normal Consistency": torch.tensor(normal_consistency).mean().item()
        })
        per_view_dict = {
            "Normal Consistency": {img : normal_consis for normal_consis, img in zip(torch.tensor(normal_consistency).tolist(), range(len(views))) }
        }

        json_path = os.path.join(model_path, name, "ours_{}".format(iteration))
        with open(json_path + "/results.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(json_path + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)

        # rendering = linear_to_srgb(render_pkg["render"])
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)      
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, normal : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.idiv, dataset.ref, dataset.deg_view)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            if normal:
                normal_bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
                render_set_normal(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, normal_bg)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            if normal:
                normal_bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
                render_set_normal(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, normal_bg)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--normal", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.normal)
