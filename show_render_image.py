"""
获取指定相机视角下的渲染图像
"""

import os
import torch
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams  # 引入所需的参数类
from argparse import ArgumentParser, Namespace
import uuid
import sys
import cv2
import numpy as np
from scene.cameras import Camera


def render_image(gaussians, pipe, background, camera):
    # 使用提供的相机参数进行渲染
    render_pkg = render(camera, gaussians, pipe, background)
    rendered_image = render_pkg["render"]
    depth_map = render_pkg["depth"]

    # 检查和转换为 numpy 数组
    rendered_image = rendered_image.detach().cpu().numpy()
    depth_map = depth_map.detach().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # 归一化深度图

    # 转换数据类型为 uint8
    rendered_image = (rendered_image * 255).astype(np.uint8)
    depth_map = (depth_map * 255).astype(np.uint8)

    # 转换为 HWC 格式
    rendered_image = np.transpose(rendered_image, (1, 2, 0))
    depth_map = np.transpose(depth_map, (1, 2, 0))

    # 转换颜色通道从 RGB 到 BGR
    rendered_image = rendered_image[..., ::-1]
    depth_map = depth_map[..., ::-1]

    # 返回渲染图像和深度图
    return rendered_image, depth_map


if __name__ == "__main__":
    parser = ArgumentParser(description="Dataset loading")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    # parser.add_argument('--debug_from', type=int, default=-1)
    # parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    # 假设已设置 PLY 文件的路径
    ply_file_path = r"D:\nizhangdaye\PycharmProjects\gaussian-splatting\output\red_house_50_depth\point_cloud\iteration_3000\point_cloud.ply"  # 替换为实际的 PLY 文件路径

    # 获取数据集参数
    dataset_params = lp.extract(parser.parse_args())

    # 创建 GaussianModel 和 Scene
    gaussians = GaussianModel(dataset_params.sh_degree)
    scene = Scene(dataset_params, gaussians, shuffle=False)

    # 从 PLY 文件加载数据
    gaussians.load_ply(ply_file_path)

    # 设置背景颜色
    bg_color = [1, 1, 1] if dataset_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 确保输出目录存在
    output_dir = r"D:\nizhangdaye\PycharmProjects\gaussian-splatting\rendered_images"
    os.makedirs(output_dir, exist_ok=True)

    # 选择一个相机进行渲染
    viewpoint_cams = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_cams[48]  # 选择第 n 个相机
    print("-----------------------")
    print(f"训练相机数量： {len(scene.getTrainCameras().copy())}")
    # print(f"viewpoint_cam 的一些参数：")
    # print(f"R: {viewpoint_cam.R}")
    # print(f"T: {viewpoint_cam.T}")
    # print(f"FoVx: {viewpoint_cam.FoVx}")
    # print(f"FoVy: {viewpoint_cam.FoVy}")

    # 渲染图像并保存
    rendered_image, depth_map = render_image(gaussians, pp.extract(args), background, viewpoint_cam)

    # 保存渲染图像
    render_map_path = os.path.join(output_dir, "rendered_image.png")
    success_render = cv2.imwrite(render_map_path, rendered_image)
    # 保存深度图
    depth_map_path = os.path.join(output_dir, "depth_map.png")
    success_depth = cv2.imwrite(depth_map_path, depth_map)

    if success_render:
        print(f"渲染图像已保存到 {render_map_path}")
    else:
        print(f"渲染图像保存失败")

    if success_depth:
        print(f"深度图已保存到 {depth_map_path}")
    else:
        print(f"深度图保存失败")
