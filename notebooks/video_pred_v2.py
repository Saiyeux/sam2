import os
# 如果使用 Apple MPS，遇到不支持的操作时回退到 CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import math
import inspect
import gc
import shutil
import re
try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False

# 解析命令行参数
parser = argparse.ArgumentParser(description="使用 SAM 2 进行视频分割，优化小区域覆盖")
parser.add_argument("--video_dir", type=str, required=True, help="包含 JPEG 帧的目录")
parser.add_argument("--output_video_path", type=str, required=True, help="输出视频文件路径")
parser.add_argument("--target_region", type=str, default="200,300,20", help="目标区域的中心和大小，格式：x,y,size")
args = parser.parse_args()

# 选择计算设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"使用设备: {device}")

# 初始化显存监控（仅限 CUDA 设备）
max_memory_used_mb = 0
max_memory_line = None
max_memory_func = None

if device.type == "cuda" and pynvml_available:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def get_memory_used_mb():
        """获取当前 GPU 显存使用量（MB），并记录最大使用时的行号和函数名"""
        global max_memory_used_mb, max_memory_line, max_memory_func
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        used_mb = mem_info.used / 1024 / 1024
        if used_mb > max_memory_used_mb:
            max_memory_used_mb = used_mb
            frame = inspect.currentframe().f_back
            max_memory_line = frame.f_lineno
            max_memory_func = frame.f_code.co_name
            print(f"显存使用量增加，当前: {used_mb:.2f} MB")
        return used_mb
else:
    def get_memory_used_mb():
        """非 CUDA 设备的占位函数"""
        return 0

def clear_cuda_cache():
    """清理 CUDA 缓存并执行垃圾回收"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    get_memory_used_mb()

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\n对 MPS 设备的支持是初步的。SAM 2 使用 CUDA 训练，在 MPS 上可能产生不同的数值输出或性能下降。"
    )

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
get_memory_used_mb()
clear_cuda_cache()

def apply_mask_to_image(image, mask, obj_id=None, random_color=False):
    """将分割掩码应用于图像"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if isinstance(image, Image.Image):
        image = np.array(image)
    masked_image = image.astype(np.float32) / 255.0
    masked_image = masked_image * (1 - mask_image[..., 3:]) + mask_image[..., :3] * mask_image[..., 3:]
    return (masked_image * 255).astype(np.uint8)

def generate_dense_points(center_x, center_y, size):
    """在指定区域内生成密集的点击点，覆盖小区域"""
    half_size = size // 2
    points = []
    # 在 size x size 的网格中生成 10 个点击点（更密集）
    step = half_size // 2
    for dx in [-half_size, -step, 0, step, half_size]:
        for dy in [-half_size, -step, 0, step, half_size]:
            if dx == 0 and dy == 0:
                continue
            points.append([center_x + dx, center_y + dy])
    points.append([center_x, center_y])  # 确保中心点包含
    return np.array(points[:10], dtype=np.float32)  # 限制最多 10 个点

# 解析目标区域参数
try:
    center_x, center_y, size = map(int, args.target_region.split(","))
    if size < 10:
        raise ValueError("目标区域大小不能小于 10 像素")
except ValueError as e:
    raise ValueError(f"无效的目标区域格式，需为 x,y,size，例如 200,300,20，错误: {e}")

# 生成边界框
box = np.array([center_x - size//2, center_y - size//2, center_x + size//2, center_y + size//2], dtype=np.float32)

# 扫描视频帧文件
frame_names = [
    p for p in os.listdir(args.video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(re.search(r'\d+', os.path.splitext(p)[0]).group()))

# 定义分块大小
block_size = 200
num_blocks = math.ceil(len(frame_names) / block_size)
temp_video_paths = []

for block_idx in range(num_blocks):
    start_idx = block_idx * block_size
    end_idx = min((block_idx + 1) * block_size, len(frame_names))
    block_frame_names = frame_names[start_idx:end_idx]
    print(f"处理分块 {block_idx + 1}/{num_blocks}（帧 {start_idx} 到 {end_idx - 1}）")

    # 为当前分块创建临时帧目录
    temp_frame_dir = f"./temp_frames_block_{block_idx}"
    os.makedirs(temp_frame_dir, exist_ok=True)
    for i, frame_name in enumerate(block_frame_names):
        src_path = os.path.join(args.video_dir, frame_name)
        dst_path = os.path.join(temp_frame_dir, f"{i}.jpg")
        os.symlink(src_path, dst_path)

    # 初始化推理状态
    inference_state = predictor.init_state(video_path=temp_frame_dir)
    predictor.reset_state(inference_state)
    get_memory_used_mb()
    clear_cuda_cache()

    ann_frame_idx = 0
    ann_obj_id = 1

    # 在第 0 帧添加密集正向点击点和边界框
    points = generate_dense_points(center_x, center_y, size)
    labels = np.ones(len(points), dtype=np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        box=box
    )
    # 保存第 0 帧掩码以便调试
    if 0 in range(start_idx, end_idx):
        mask = (out_mask_logits[0] > -3.0).cpu().numpy()
        plt.imsave(f"mask_frame_0_block_{block_idx}.png", mask)
    del out_mask_logits
    get_memory_used_mb()
    clear_cuda_cache()

    # 运行传播
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        global_frame_idx = start_idx + out_frame_idx
        if start_idx <= global_frame_idx < end_idx:
            mask = (out_mask_logits[0] > -3.0).cpu().numpy()
            video_segments[global_frame_idx] = {out_obj_id: mask}
        del out_mask_logits
        get_memory_used_mb()
        clear_cuda_cache()

    # 在第 150 帧进行精炼（如果在当前分块内）
    if start_idx <= 150 < end_idx:
        ann_frame_idx = 150 - start_idx
        # 添加更多负向点击点以排除噪声
        points = np.array([
            [center_x + size, center_y + size],
            [center_x - size, center_y - size],
            [center_x + size, center_y - size],
            [center_x - size, center_y + size],
        ], dtype=np.float32)
        labels = np.zeros(len(points), dtype=np.int32)
        _, _, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels
        )
        # 保存第 150 帧掩码以便调试
        mask = (out_mask_logits[0] > -3.0).cpu().numpy()
        plt.imsave(f"mask_frame_150_block_{block_idx}.png", mask)
        del out_mask_logits
        get_memory_used_mb()
        clear_cuda_cache()

        # 再次运行传播以更新结果
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            global_frame_idx = start_idx + out_frame_idx
            if start_idx <= global_frame_idx < end_idx:
                mask = (out_mask_logits[0] > -3.0).cpu().numpy()
                video_segments[global_frame_idx] = {out_obj_id: mask}
            del out_mask_logits
            get_memory_used_mb()
            clear_cuda_cache()

    # 创建临时视频
    temp_video_path = f"{args.output_video_path}_part{block_idx}.mp4"
    temp_video_paths.append(temp_video_path)
    writer = imageio.get_writer(temp_video_path, fps=30, codec='libx264')

    # 处理分块中的每帧
    for frame_idx in range(start_idx, end_idx):
        frame_path = os.path.join(args.video_dir, frame_names[frame_idx])
        frame = Image.open(frame_path)
        
        if frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[frame_idx].items():
                frame = apply_mask_to_image(frame, out_mask, obj_id=out_obj_id)
        else:
            frame = np.array(frame)
        
        writer.append_data(frame)
        get_memory_used_mb()
        clear_cuda_cache()

    writer.close()
    print(f"临时视频保存至 {temp_video_path}")

    # 清理临时帧目录和变量
    shutil.rmtree(temp_frame_dir)
    del video_segments
    get_memory_used_mb()
    clear_cuda_cache()

# 合并所有临时视频
final_writer = imageio.get_writer(args.output_video_path, fps=30, codec='libx264')
for temp_video_path in temp_video_paths:
    reader = imageio.get_reader(temp_video_path)
    for frame in reader:
        final_writer.append_data(frame)
    reader.close()
    os.remove(temp_video_path)
    get_memory_used_mb()
    clear_cuda_cache()
final_writer.close()

# 清理 pynvml
if device.type == "cuda" and pynvml_available:
    pynvml.nvmlShutdown()

print(f"最终视频保存至 {args.output_video_path}")
print(f"最大显存使用量: {max_memory_used_mb:.2f} MB")
if max_memory_line and max_memory_func:
    print(f"最大显存发生在行 {max_memory_line}，函数 {max_memory_func}")
else:
    print("未记录最大显存的行号/函数（非 CUDA 设备或 pynvml 不可用）")