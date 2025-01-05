import os
import subprocess

# 视频绝对路径
video_path = r"D:\nizhangdaye\PycharmProjects\gaussian-splatting\data\huang_long\杭州黄龙体育中心_哔哩哔哩_bilibili.mp4"
# 切分帧数，每秒多少帧
fps = 2

# 获取当前工作路径
current_path = os.getcwd()
# 上一级文件夹所在路径
folder_path = os.path.dirname(video_path)
# 图片保存路径
images_path = os.path.join(folder_path, 'input')
os.makedirs(images_path, exist_ok=True)

ffmpeg_path = os.path.join(current_path, 'external', r'ffmpeg/bin/ffmpeg.exe')

# 脚本运行
# 视频切分脚本
command = f'{ffmpeg_path} -i {video_path} -qscale:v 1 -qmin 1 -vf fps={fps} {images_path}\\%04d.jpg'
subprocess.run(command, shell=True)
# COLMAP估算相机位姿
command = f'python convert.py -s {folder_path}'
subprocess.run(command, shell=True)
# 模型训练脚本，模型会保存在output路径下
command = f'python train.py -s {folder_path}'
subprocess.run(command, shell=True)
