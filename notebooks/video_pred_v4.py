import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import cv2
import pynvml
from datetime import datetime
from PIL import Image
import argparse
import tempfile
import shutil
from sam2.build_sam import build_sam2_video_predictor

# 解析命令行参数
parser = argparse.ArgumentParser(description="使用SAM 2进行视频分割，直接使用指定边界框作为提示")
parser.add_argument("--input_dir", type=str, required=True, help="包含输入视频帧（JPEG）的目录")
parser.add_argument("--output_path", type=str, required=True, help="保存输出视频（MP4）和边界框图像的目录")
parser.add_argument("--box", type=float, nargs=4, metavar=('x1', 'y1', 'x2', 'y2'), 
                    required=True, help="指定边界框坐标 [x1, y1, x2, y2]，例如 --box 0 540 1440 1080")
args = parser.parse_args()

video_dir = args.input_dir
output_base_dir = args.output_path
user_box = args.box  # 用户指定的边界框 [x1, y1, x2, y2]

# 初始化pynvml用于GPU内存监控
def init_pynvml():
    try:
        pynvml.nvmlInit()
        return True
    except pynvml.NVMLError:
        print("无法初始化pynvml，GPU内存监控已禁用。")
        return False

# 获取GPU内存使用情况
def get_gpu_memory(device_idx=0):
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1024**2, mem_info.total / 1024**2  # 返回MB单位
    except pynvml.NVMLError:
        return None, None

# 验证边界框坐标
def validate_bounding_box(box, image_shape):
    """
    验证边界框坐标是否有效。

    Args:
        box: List[float], 边界框坐标 [x1, y1, x2, y2]
        image_shape: Tuple, 图像形状 (height, width, channels)

    Returns:
        List[int], 修正后的边界框坐标 [x1, y1, x2, y2]

    Raises:
        ValueError: 如果边界框无效
    """
    height, width = image_shape[:2]
    x1, y1, x2, y2 = map(float, box)

    # 检查坐标范围
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"边界框 [{x1}, {y1}, {x2}, {y2}] 超出图像范围 [0, 0, {width}, {height}] 或无效")
    
    # 检查逻辑有效性
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise ValueError("边界框宽度或高度过小")

    return [int(x1), int(y1), int(x2), int(y2)]

# 在第一帧上绘制边界框并保存
def save_box_annotated_frame(frame, box, output_path):
    """
    在帧上绘制边界框并保存为图像。

    Args:
        frame: np.ndarray, 输入帧（RGB格式）
        box: List[int], 边界框坐标 [x1, y1, x2, y2]
        output_path: str, 输出目录
    """
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = box
    
    # 绘制边界框（红色，厚度2）
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 标注坐标（白色，字体大小0.5）
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_bgr, f"({x1},{y1})", (x1, y1 - 10), font, 0.5, (255, 255, 255), 1)
    cv2.putText(frame_bgr, f"({x2},{y2})", (x2, y2 + 20), font, 0.5, (255, 255, 255), 1)
    
    # 保存图像
    output_file = os.path.join(output_path, "box_annotated_first_frame.png")
    cv2.imwrite(output_file, frame_bgr)
    print(f"已保存带边界框的首帧图像: {output_file}")

# 创建掩码图像
def create_mask_image(mask, obj_id=None, random_color=False):
    import matplotlib.pyplot as plt
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return (mask_image * 255).astype(np.uint8)

# 在帧上叠加掩码
def overlay_mask_on_frame(frame, mask_image):
    frame = np.array(frame)
    mask_image = mask_image[:, :, :3]
    overlay = cv2.addWeighted(frame, 0.7, mask_image, 0.3, 0)
    return overlay

# 选择计算设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"使用设备: {device}")

# 如果使用CUDA，初始化pynvml
pynvml_initialized = init_pynvml() if device.type == "cuda" else False

# 配置CUDA设备优化
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    used_mb, total_mb = get_gpu_memory()
    print(f"初始GPU内存使用量: {used_mb:.2f}/{total_mb:.2f} MB")
elif device.type == "mps":
    print(
        "\nMPS设备支持为初步支持。SAM 2在CUDA上训练，可能在MPS上产生不同输出或性能下降。"
    )

# 加载SAM 2视频预测器
sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

print("加载SAM 2视频预测器...")
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
print("SAM 2视频预测器加载完成。")

# 扫描所有JPEG帧文件名
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
print(f"在 {video_dir} 中找到 {len(frame_names)} 帧")

# 初始化视频写入器参数
first_frame = Image.open(os.path.join(video_dir, frame_names[0]))
width, height = first_frame.size
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
FPS = 30.0

# 创建输出目录
os.makedirs(output_base_dir, exist_ok=True)
print(f"输出目录创建/验证: {output_base_dir}")

# 创建临时目录
temp_dir = tempfile.mkdtemp()
try:
    print(f"将 {len(frame_names)} 帧复制到临时目录 {temp_dir}...")
    for frame_idx, frame_name in enumerate(frame_names):
        src_path = os.path.join(video_dir, frame_name)
        dst_path = os.path.join(temp_dir, f"{frame_idx}.jpg")
        shutil.copy(src_path, dst_path)
    print("帧复制完成。")
    
    # 在初始化状态前清除缓存
    if device.type == "cuda":
        torch.cuda.empty_cache()
        used_mb, total_mb = get_gpu_memory()
        print(f"初始化状态前的GPU内存: {used_mb:.2f}/{total_mb:.2f} MB")
    
    # 使用临时目录初始化推理状态
    print("初始化推理状态...")
    inference_state = predictor.init_state(video_path=temp_dir)
    predictor.reset_state(inference_state)
    print("推理状态初始化完成。")
    
    # 初始化后清除缓存
    if device.type == "cuda":
        torch.cuda.empty_cache()
        used_mb, total_mb = get_gpu_memory()
        print(f"初始化状态后的GPU内存: {used_mb:.2f}/{total_mb:.2f} MB")
    
    # 加载首帧以验证边界框和绘制
    print("加载首帧以验证边界框...")
    first_frame_path = os.path.join(temp_dir, "0.jpg")
    first_frame_img = np.array(Image.open(first_frame_path).convert("RGB"))
    
    # 验证用户指定的边界框
    try:
        optimal_box = validate_bounding_box(user_box, first_frame_img.shape)
        print(f"使用用户指定边界框: {optimal_box}")
    except ValueError as e:
        print(f"错误：{e}")
        raise
    
    # 在首帧上绘制边界框并保存
    print("保存首帧的边界框图像...")
    save_box_annotated_frame(first_frame_img, optimal_box, output_base_dir)
    
    # 为视频预测器添加边界框提示
    obj_id = 1
    print(f"在帧0为对象 {obj_id} 添加边界框提示")
    box = np.array(optimal_box, dtype=np.float32)  # [x1, y1, x2, y2]
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        box=box,
    )
    print(f"已处理对象 {obj_id} 的边界框")
    
    # 添加边界框后清除缓存
    if device.type == "cuda":
        torch.cuda.empty_cache()
        used_mb, total_mb = get_gpu_memory()
        print(f"添加边界框后的GPU内存: {used_mb:.2f}/{total_mb:.2f} MB")
    
    # 在所有帧上进行传播
    video_segments = {}
    print("开始视频传播...")
    start_time = datetime.now()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        if out_frame_idx % 50 == 0:
            print(f"已处理帧 {out_frame_idx}/{len(frame_names) - 1}")
            if device.type == "cuda":
                used_mb, total_mb = get_gpu_memory()
                print(f"帧 {out_frame_idx} 的GPU内存: {used_mb:.2f}/{total_mb:.2f} MB")
    print(f"视频传播完成，耗时 {datetime.now() - start_time}")
    
    # 传播后清除缓存
    if device.type == "cuda":
        torch.cuda.empty_cache()
        used_mb, total_mb = get_gpu_memory()
        print(f"传播后的GPU内存: {used_mb:.2f}/{total_mb:.2f} MB")
    
    # 初始化视频写入器
    output_video_path = os.path.join(output_base_dir, "output_video.mp4")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, FPS, (width, height))
    print(f"初始化视频写入器: {output_video_path}")
    
    # 生成视频
    print("生成输出视频...")
    for frame_idx in range(len(frame_names)):
        frame = Image.open(os.path.join(video_dir, frame_names[frame_idx]))
        frame_np = np.array(frame)
        
        # 创建组合掩码
        combined_mask = np.zeros((height, width, 4), dtype=np.uint8)
        if frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[frame_idx].items():
                mask_image = create_mask_image(out_mask, obj_id=out_obj_id % 10)
                combined_mask = np.maximum(combined_mask, mask_image)
        
        # 叠加掩码
        overlay_frame = overlay_mask_on_frame(frame_np, combined_mask)
        
        # 写入视频
        overlay_frame_bgr = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(overlay_frame_bgr)
        
        if frame_idx % 50 == 0:
            print(f"已将帧 {frame_idx}/{len(frame_names) - 1} 写入视频")
    
    # 释放视频写入器
    video_writer.release()
    print(f"输出视频已保存: {output_video_path}")
    
    # 清除推理状态和缓存
    del inference_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
        used_mb, total_mb = get_gpu_memory()
        print(f"清理后的GPU内存: {used_mb:.2f}/{total_mb:.2f} MB")

finally:
    # 清理临时目录
    print(f"删除临时目录 {temp_dir}...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("临时目录已删除。")

# 清理pynvml
if pynvml_initialized:
    pynvml.nvmlShutdown()

# 最终缓存清理
if device.type == "cuda":
    torch.cuda.empty_cache()
    used_mb, total_mb = get_gpu_memory()
    print(f"最终GPU内存使用量: {used_mb:.2f}/{total_mb:.2f} MB")