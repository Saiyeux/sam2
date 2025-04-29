import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
import logging
import json

# 如果使用 Apple MPS，遇到不支持的操作时回退到 CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def setup_device():
    """设置计算设备并进行优化配置"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(
            "\n警告: MPS 设备支持是初步的，SAM2 在 CUDA 上训练，可能在 MPS 上性能下降或结果不同。\n"
            "详情见: https://github.com/pytorch/pytorch/issues/84936"
        )
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")
    return device

def get_complementary_color(color):
    """计算颜色的反色（补色），输入和输出为RGB格式（0-1范围）"""
    return 1 - color[:3]

def show_anns(anns, borders=True, tracking_points=None, top_n=None):
    """可视化分割掩码和追踪点，追踪点使用区域颜色的反色"""
    if not anns:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0

    region_colors = []

    for idx, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        region_colors.append(color_mask)
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

    if tracking_points and top_n is not None:
        for idx, region_points in enumerate(tracking_points[:top_n]):
            if idx >= len(region_colors):
                break
            region_color = region_colors[idx]
            complementary_color = get_complementary_color(region_color)
            complementary_color = tuple(complementary_color)
            for x, y in region_points:
                plt.plot(x, y, marker='o', color=complementary_color, markersize=2)

def get_memory_usage():
    """获取当前 GPU 显存占用（MB），若无 GPU 返回 0"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0, 0

def sample_points_from_mask(mask, x_interval=10, y_interval=10, border_reduction_factor=10):
    """从掩码中按指定间隔采样点，减少边界点，并去除每行开头和结尾的点"""
    # 识别边界
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    border_mask = mask - eroded_mask

    # 初始采样点
    points = []
    h, w = mask.shape
    border_interval_x = x_interval * border_reduction_factor
    border_interval_y = y_interval * border_reduction_factor

    for y in range(0, h, y_interval):
        for x in range(0, w, x_interval):
            if border_mask[y, x] > 0:
                if (x % border_interval_x == 0) and (y % border_interval_y == 0):
                    points.append((x, y))
            elif mask[y, x] > 0:
                points.append((x, y))

    # 按行分组，去除每行开头和结尾的点
    if not points:
        return points

    # 按 y 坐标分组
    points_by_row = {}
    for x, y in points:
        if y not in points_by_row:
            points_by_row[y] = []
        points_by_row[y].append((x, y))

    # 去除每行开头和结尾的点
    filtered_points = []
    for y, row_points in points_by_row.items():
        if len(row_points) <= 2:  # 如果一行只有1-2个点，直接跳过（否则全被移除）
            continue
        # 按 x 坐标排序
        row_points.sort(key=lambda p: p[0])
        # 移除开头和结尾的点
        trimmed_points = row_points[5:-3]
        filtered_points.extend(trimmed_points)

    return filtered_points

def process_image(image_path, mask_generator, output_dir, logger, sample_interval=10, top_n=5):
    """处理单张图片，分割区域并为面积最大的前 top_n 个区域生成追踪点"""
    logger.info(f"正在处理图片: {image_path}")
    
    start_time = time.time()
    mem_before_alloc, mem_before_res = get_memory_usage()
    logger.info(f"推理前显存 - 分配: {mem_before_alloc:.2f} MB, 保留: {mem_before_res:.2f} MB")

    # 加载图像
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    # 生成 SAM2 掩码
    masks = mask_generator.generate(image)
    logger.info(f"生成掩码数量: {len(masks)}")

    # 按面积排序掩码
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # 限制 top_n 不超过掩码数量
    top_n = min(top_n, len(sorted_masks))
    logger.info(f"选取面积最大的前 {top_n} 个区域生成追踪点")

    # 为前 top_n 个区域生成追踪点
    tracking_points_per_region = []
    for idx, mask in enumerate(sorted_masks[:top_n]):
        segmentation = mask['segmentation'].astype(np.uint8) * 255
        region_points = sample_points_from_mask(segmentation, x_interval=sample_interval, y_interval=sample_interval)
        tracking_points_per_region.append(region_points)
        logger.info(f"区域 {idx} 追踪点数量: {len(region_points)}")

    # 保存追踪点坐标为 JSON
    tracking_data = {
        "image": os.path.basename(image_path),
        "regions": [
            {"region_id": idx, "tracking_points": [list(p) for p in points]}
            for idx, points in enumerate(tracking_points_per_region)
        ]
    }
    coords_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_tracking_points.json")
    with open(coords_filename, 'w') as f:
        json.dump(tracking_data, f, indent=4)
    logger.info(f"追踪点坐标已保存至: {coords_filename}")

    # 可视化并保存分割结果
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(sorted_masks, tracking_points=tracking_points_per_region, top_n=top_n)
    plt.axis('off')

    output_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_segmented.png")
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    logger.info(f"结果已保存至: {output_filename}")

    mem_after_alloc, mem_after_res = get_memory_usage()
    logger.info(f"推理后显存 - 分配: {mem_after_alloc:.2f} MB, 保留: {mem_after_res:.2f} MB")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("已清空缓存")

    process_time = time.time() - start_time
    logger.info(f"单图处理时间: {process_time:.2f} 秒")

    return tracking_points_per_region

def main():
    """主函数：处理命令行参数，执行图片分割并生成追踪点"""
    np.random.seed(3)

    log_file = os.path.join(os.getcwd(), "segmentation_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("\n")
    logger.info("\n" + '-' * 10 + " Section starts here " + '-' * 10)
    logger.info(f"日志文件保存至: {log_file}")

    parser = argparse.ArgumentParser(description="使用 SAM2 对图片进行自动分割并生成追踪点")
    parser.add_argument('--input_dir', type=str, default='/home/surgicalai/Data/images_3', help="输入图片目录路径")
    parser.add_argument('--output_dir', type=str, default="/home/surgicalai/Data/output/sam2", help="输出图片保存目录")
    parser.add_argument('--sample_interval', type=int, default=30, help="追踪点的x和y间隔（像素）")
    parser.add_argument('--top_n', type=int, default=3, help="选取面积最大的前n个区域生成追踪点")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"输出目录: {args.output_dir}")

    device = setup_device()

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    logger.info("正在加载 SAM2 模型...")
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0,
        use_m2m=True,
    )
    logger.info("模型加载完成")

    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(image_extensions)]
    if not image_files:
        logger.error(f"错误: 输入目录 {args.input_dir} 中未找到图片文件")
        sys.exit(1)

    logger.info(f"找到 {len(image_files)} 张图片，开始处理...")
    
    total_start_time = time.time()
    
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(args.input_dir, image_file)
        logger.info(f"\n[{idx}/{len(image_files)}] 开始处理...")
        process_image(image_path, mask_generator, args.output_dir, logger, sample_interval=args.sample_interval, top_n=args.top_n)

    total_time = time.time() - total_start_time
    logger.info(f"\n所有图片处理完成！总推理时间: {total_time:.2f} 秒")
    logger.info('-' * 10 + " Section ends here " + '-' * 12 + "\n")
    logger.info("\n")

if __name__ == "__main__":
    main()