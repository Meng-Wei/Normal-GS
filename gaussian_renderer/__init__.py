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

import math
import torch
from einops import repeat
from diff_gaussian_rasterization import GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings

from scene.gaussian_model import GaussianModel
from utils.ref_utils import reflect
from utils.general_utils import get_minimum_axis, flip_align_view

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel,
                              visible_mask=None, is_training=False,
                              render_n=False, render_dotprod=False):
    ## view frustum filtering for acceleration
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    offsets = grid_offsets.view([-1, 3]) # [mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    ## view-adaptive multi-resolution feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        appearance = pc.get_appearance(camera_indicies)

    neural_opacity = pc.get_opacity_mlp(cat_local_view) if pc.add_opacity_dist else pc.get_opacity_mlp(cat_local_view_wodist)
    scale_rot = pc.get_cov_mlp(cat_local_view) if pc.add_cov_dist else pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # get offset's color, reuse as the albedo if IDIV enabled
    local_feature = cat_local_view if pc.add_color_dist else cat_local_view_wodist
    if pc.appearance_dim > 0:
        local_feature = torch.cat([local_feature, appearance], dim=1)
    color = pc.get_color_mlp(local_feature)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3]) # [mask]

    local_feature = cat_local_view
    if pc.enable_idiv: # idiv
        # idiv = pc.get_idiv_mlp(local_feature)
        idiv = pc.get_idiv_mlp(cat_local_view)
        idiv = idiv.reshape([anchor.shape[0]*pc.n_offsets, 3])

    if pc.enable_ref: # tint, roughness
        tint = pc.get_tint_mlp(local_feature)
        tint = tint.reshape([anchor.shape[0]*pc.n_offsets, 3]) # [mask]
        roughness = pc.get_roughness_mlp(local_feature)
        roughness = roughness.reshape([anchor.shape[0]*pc.n_offsets, 1]) # [mask]

    # local_feature = cat_local_view
    # concatenate and filter by mask
    if pc.enable_ref:
        concatenated = torch.cat([local_feature, grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets, tint, roughness], dim=-1)
    else:
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    if pc.enable_idiv: # Concatenate IDIV
        concatenated_all = torch.cat([concatenated_all, idiv], dim=-1)
    masked = concatenated_all[mask]
    opacity = neural_opacity[mask]

    # split attributes
    if pc.enable_idiv and pc.enable_ref:
        features, scaling_repeat, repeat_anchor, color, scale_rot, offsets, tint, roughness, idiv = masked.split([pc.color_dim+pc.appearance_dim + 1, 6, 3, 3, 7, 3, 3, 1, 3], dim=-1)
    elif pc.enable_ref:
        features, scaling_repeat, repeat_anchor, color, scale_rot, offsets, tint, roughness = masked.split([pc.color_dim+pc.appearance_dim + 1, 6, 3, 3, 7, 3, 3, 1], dim=-1)
    elif pc.enable_idiv:
        scaling_repeat, repeat_anchor, color, scale_rot, offsets, idiv = masked.split([6, 3, 3, 7, 3, 3], dim=-1)
    else:
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    # post-process normals
    # repeat_view = repeat(ob_view, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    # repeat_view = repeat_view[mask]
    if render_n or pc.enable_ref or pc.enable_idiv:
        normal = get_minimum_axis(scaling, rot) # -1, 1; normalized. global, from object outwards
        dir_pp = xyz - viewpoint_camera.camera_center.repeat(xyz.shape[0], 1)
        dir_pp_normalized = (dir_pp / dir_pp.norm(dim=1, keepdim=True)).detach() # from camera to object
        normal, _ = flip_align_view(normal, dir_pp_normalized)
        dotprod = torch.sum(normal * -dir_pp_normalized, dim=-1, keepdims=True)
        # normal = get_minimum_axis(scaling, rot) # -1, 1; normalized. global, from object outwards
        # normal, _ = flip_align_view(normal, repeat_view)
        # dotprod = torch.sum(normal * -repeat_view, dim=-1, keepdims=True)

    out = {"xyz": xyz, "opacity": opacity, "scaling": scaling, "rot": rot}
    if is_training:
        out.update({"neural_opacity": neural_opacity, "mask": mask})
    if render_dotprod: # Used to penalize back-facing normals
        reg_dotprod = torch.clamp(-dotprod, 0.0) ** 2
        out.update({"dotprod": reg_dotprod})
    if render_n:
        # out.update({"normal": normal}) # -1, 1; normalized. global, from object outwards
        local_normal = normal @ viewpoint_camera.world_view_transform[:3, :3]
        out.update({"normal": local_normal})

    # post-process color
    diffuse_color = color # Original or Albedo
    if pc.enable_idiv:
        mid_val = torch.sum(idiv * normal, dim=-1, keepdims=True).abs()
        diffuse_color = mid_val * diffuse_color

    specular_color = 0
    if pc.enable_ref:
        # integrated directional embedding
        reflect_dir = reflect(-dir_pp_normalized, normal)
        # reflect_dir = reflect(-repeat_view, normal)
        ide = pc.ide_fn(reflect_dir, roughness)

        specular_color = pc.get_specular_mlp(torch.cat([ide, dotprod, features], dim=-1))
        specular_color = specular_color.reshape([-1, 3])
        specular_color = specular_color

    out.update({"color": specular_color + diffuse_color})
    return out

def render(viewpoint_camera, pc : GaussianModel, pipe,
           bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None,
           retain_grad=False, render_n=False, render_dotprod=False,
           render_full=False, down_sampling=1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Fetch Gaussian attributes
    is_training = pc.get_color_mlp.training
    neural_gaussians = generate_neural_gaussians(viewpoint_camera, pc,
        visible_mask, is_training, render_n, render_dotprod)
    xyz = neural_gaussians["xyz"]
    color = neural_gaussians["color"]
    opacity = neural_gaussians["opacity"]
    scaling = neural_gaussians["scaling"]
    rot = neural_gaussians["rot"]
    if is_training:
        neural_opacity = neural_gaussians["neural_opacity"]
        mask = neural_gaussians["mask"]

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    image_height = int(viewpoint_camera.image_height * down_sampling)
    image_width = int(viewpoint_camera.image_width * down_sampling)

    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Set up rasterization without background colors
    raster_settings_nobg = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )
    rasterizer_nobg = GaussianRasterizer(raster_settings=raster_settings_nobg)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    outputs = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    rendered_image = outputs[0]
    radii = outputs[1]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    out = {"render": rendered_image,
           "viewspace_points": screenspace_points,
           "visibility_filter" : radii > 0,
           "radii": radii}

    # Render other attributes: alpha, depth, normals, or dotprod
    # TODO: Slow implementation
    if render_full:
        alpha = torch.ones_like(xyz)
        out["alpha"] =  rasterizer_nobg(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = alpha,
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)[0]

        p_hom = torch.cat([xyz, torch.ones_like(xyz[...,:1])], -1).unsqueeze(-1)
        p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0,1), p_hom)
        p_view = p_view[...,:3,:]
        depth = p_view.squeeze()[...,2:3]
        depth = depth.repeat(1,3)
        out["depth"] = rasterizer_nobg(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = depth,
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)[0]

    if is_training:
        out.update({"selection_mask": mask,
                    "neural_opacity": neural_opacity,
                    "scaling": scaling})

    if render_n:
        # normal: normalized, -1, 1
        normal = neural_gaussians["normal"]
        normal_image, _ = rasterizer_nobg(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = normal * 0.5 + 0.5, # [-1, 1] to [0, 1]
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)
        normal_image = (normal_image - 0.5) * 2 # [0, 1] to [-1, 1]
        out.update({"normal" : normal_image})
    return out

def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
