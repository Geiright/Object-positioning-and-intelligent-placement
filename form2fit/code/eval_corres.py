"""Run the Form2Fit models on the benchmark.
"""

import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm
from scipy.stats import mode

from form2fit import config
from form2fit.code.ml.models import CorrespondenceNet
from form2fit.code.ml.dataloader import get_corr_loader
from form2fit.code.utils import misc, ml
from form2fit.code.utils.pointcloud import transform_xyz

from walle.core import Pose, RotationMatrix


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    save_dir = os.path.join("./dump/")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dloader = get_corr_loader(
        args.foldername,
        batch_size=1,
        sample_ratio=1,
        shuffle=False,
        dtype=args.dtype,
        num_rotations=20,
        num_workers=1,
        markovian=True,
        augment=False,
        background_subtract=config.BACKGROUND_SUBTRACT[args.foldername],
    )

    stats = dloader.dataset.stats
    color_mean = stats[0][0]
    color_std = stats[0][1]
    depth_mean = stats[1][0]
    depth_std = stats[1][1]

    # load model
    model = CorrespondenceNet(2, 64, 20).to(device)
    if args.use_official:
        state_dict = torch.load(os.path.join(config.weights_dir, "matching", args.foldername + ".tar"), map_location=device)
        model.load_state_dict(state_dict['model_state'])
    else:
        state_dict = torch.load(os.path.join(config.weights_dir,"corres-epoch1680"+".pth"),map_location=device)#TODO:要换一个名字
        # model.load_state_dict(state_dict)
        model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
    model.eval()

    estimated_poses = []
    for idx, (imgs, labels, center) in enumerate(dloader):
        print("{}/{}".format(idx+1, len(dloader)))

        imgs, labels = imgs.to(device), labels.to(device)

        # remove padding from labels
        label = labels[0]
        mask = torch.all(label == torch.LongTensor([999]).repeat(6).to(device), dim=1)
        label = label[~mask]

        # extract correspondences from label
        source_idxs = label[:, 0:2]
        target_idxs = label[:, 2:4]
        rot_idx = label[:, 4]
        is_match = label[:, 5]
        correct_rot = rot_idx[0]
        mask = (is_match == 1) & (rot_idx == correct_rot)
        kit_idxs = source_idxs[mask]
        obj_idxs = target_idxs[mask]
        kit_idxs_all = kit_idxs.clone()
        obj_idxs_all = obj_idxs.clone()
        if args.subsample is not None:
            kit_idxs = kit_idxs[::int(args.subsample)]
            obj_idxs = obj_idxs[::int(args.subsample)]

        H, W = imgs.shape[2:]

        # compute kit and object descriptor maps
        with torch.no_grad():
            outs_s, outs_t = model(imgs, *center[0])
        out_s = outs_s[0]
        D, H, W = outs_t.shape[1:]
        out_t = outs_t[0:1]
        out_t_flat = out_t.view(1, D, H * W).permute(0, 2, 1)

        color_kit = ml.tensor2ndarray(imgs[:, 0, :], [color_mean, color_std], True).squeeze()
        color_obj = ml.tensor2ndarray(imgs[:, 2, :], [color_mean, color_std], True).squeeze()
        depth_kit = ml.tensor2ndarray(imgs[:, 1, :], [depth_mean, depth_std], False).squeeze()
        depth_obj = ml.tensor2ndarray(imgs[:, 3, :], [depth_mean, depth_std], False).squeeze()
        if args.debug:#第一张图,盒子逆着角度真实值旋转,然后画上盒子洞(盒子洞没有旋转?)和物体洞
            kit_idxs_np = kit_idxs_all.detach().cpu().numpy().squeeze()
            obj_idxs_np = obj_idxs_all.detach().cpu().numpy().squeeze()
            fig, axes = plt.subplots(1, 2)
            color_kit_r = misc.rotate_img(color_kit, -(360/20)*correct_rot, center=(center[0][1], center[0][0]))
            axes[0].imshow(color_kit_r)
            axes[0].scatter(kit_idxs_np[:, 1], kit_idxs_np[:, 0], c='r')#红色点画的是盒子点
            axes[1].imshow(color_obj)
            axes[1].scatter(obj_idxs_np[:, 1], obj_idxs_np[:, 0], c='b')#蓝色点画的是物体点
            for ax in axes:
                ax.axis('off')
            plt.savefig("eval_corres_1.png")
            plt.show()

        # loop through ground truth correspondences
        obj_uvs = []
        predicted_kit_uvs = []
        rotations = []
        correct = 0
        for corr_idx, (u, v) in enumerate(tqdm(obj_idxs, leave=False)):
            idx_flat = u*W + v
            target_descriptor = torch.index_select(out_t_flat, 1, idx_flat).squeeze(0)#shape=(1,64)
            outs_s_flat = out_s.view(20, out_s.shape[1], H*W)#shape=(20,64,H*W)
            target_descriptor = target_descriptor.unsqueeze(0).repeat(20, H*W, 1).permute(0, 2, 1)#shape=(20,64,H*W)
            diff = outs_s_flat - target_descriptor
            l2_dist = diff.pow(2).sum(1).sqrt()
            heatmaps = l2_dist.view(l2_dist.shape[0], H, W).cpu().numpy()
            predicted_best_idx = l2_dist.min(dim=1)[0].argmin()
            rotations.append(predicted_best_idx.item())
            if predicted_best_idx == correct_rot:
                correct += 1
            min_val = heatmaps[predicted_best_idx].argmin()
            u_min, v_min = np.unravel_index(min_val, (H, W))
            predicted_kit_uvs.append([u_min.item(), v_min.item()])
            obj_uvs.append([u.item(), v.item()])
        # print("acc: {}".format(correct / len(obj_idxs)))

        # compute rotation majority
        best_rot = mode(rotations)[0][0]

        # eliminate correspondences with rotation different than mode
        select_idxs = np.array(rotations) == best_rot
        predicted_kit_uvs = np.array(predicted_kit_uvs)[select_idxs]
        obj_uvs = np.array(obj_uvs)[select_idxs]

        # use predicted correspondences to estimate affine transformation
        src_pts = np.array(obj_uvs)[:, [1, 0]]
        dst_pts = np.array(predicted_kit_uvs)
        dst_pts = misc.rotate_uv(dst_pts, (360/20)*best_rot, H, W, cxcy=center[0])[:, [1, 0]]

        # compose transform
        zs = depth_obj[src_pts[:, 1], src_pts[:, 0]].reshape(-1, 1)
        src_xyz = np.hstack([src_pts, zs])#xy坐标和在depth_obj上对应的像素值(像素值作为z?)
        src_xyz[:, 0] = (src_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]#这是归一化的作用吗?
        src_xyz[:, 1] = (src_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
        zs = depth_kit[dst_pts[:, 1], dst_pts[:, 0]].reshape(-1, 1)
        dst_pts[:, 0] += W
        dst_xyz = np.hstack([dst_pts, zs])#(完整图中的)xy坐标和depth_kit上对应的像素值(像素值作为z)
        dst_xyz[:, 0] = (dst_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
        dst_xyz[:, 1] = (dst_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
        m1 = np.eye(4)
        dst_xyz[:, 2] = src_xyz[:, 2]#物体的z坐标值赋给盒子?
        m1[:3, 3] = np.mean(dst_xyz, axis=0) - np.mean(src_xyz, axis=0)#m1平移矩阵,表示的变换是从物体xzy坐标均值移到盒子xyz坐标均值
        m2 = np.eye(4)
        m2[:3, 3] = -np.mean(dst_xyz, axis=0)#m2平移矩阵,表示的变换是从盒子xzy坐标均值移动到原点
        m3 = np.eye(4)
        m3[:3, :3] = RotationMatrix.rotz(np.radians(-(360/20)*best_rot))#旋转矩阵,根据best_rot,以z轴为中心反方向旋转对应角度
        m4 = np.eye(4)
        m4[:3, 3] = np.mean(dst_xyz, axis=0)##m4平移矩阵,表示的变换是从原点移动到盒子xzy坐标均值
        estimated_pose = m4 @ m3 @ m2 @ m1#表示的变换:物体xyz坐标均值移到原点,逆向旋转最佳角度之后,移回盒子xyz坐标均值
        estimated_poses.append(estimated_pose)#if debug,estimated_pose将和物体mask相乘

        # plot
        if args.debug:#第二张图,物体mask在点云中根据estimated_pose变化,然后转回像素坐标,在完整的图中显示变化后的mask
            img = np.zeros((H, W*2))
            img[:, :W] = color_obj
            img[:, W:] = color_kit
            zs = depth_obj[obj_idxs_np[:, 0], obj_idxs_np[:, 1]].reshape(-1, 1)
            mask_xyz = np.hstack([obj_idxs_np, zs])
            mask_xyz[:, [0, 1]] = mask_xyz[:, [1, 0]]#坐标和物体深度图上的对应像素作为xyz
            mask_xyz[:, 0] = (mask_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
            mask_xyz[:, 1] = (mask_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
            mask_xyz = transform_xyz(mask_xyz, estimated_pose)
            mask_xyz[:, 0] = (mask_xyz[:, 0] - config.VIEW_BOUNDS[0, 0]) / config.HEIGHTMAP_RES
            mask_xyz[:, 1] = (mask_xyz[:, 1] - config.VIEW_BOUNDS[1, 0]) / config.HEIGHTMAP_RES
            hole_idxs = mask_xyz[:, [1, 0]]
            plt.imshow(img)
            plt.scatter(hole_idxs[:, 1], hole_idxs[ :, 0])
            plt.savefig("eval_corres_2.png")
            plt.show()

    with open(os.path.join(save_dir, "{}_poses-our.pkl".format(args.foldername)), "wb") as fp:#dataloder中加载的所有时间步得到的estimated_pose形成列表,保存下来
        pickle.dump(estimated_poses, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Form2Fit Matching Module on Benchmark")
    parser.add_argument("--foldername", type=str, default="fruits",help="The name of the dataset.")
    parser.add_argument("--dtype", type=str, default="test")
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("-uo","--use_official", action="store_true", help="whether to use official weights")
    parser.add_argument("--debug", type=lambda s: s.lower() in ["1", "true"], default=False)
    args, unparsed = parser.parse_known_args()
    main(args)