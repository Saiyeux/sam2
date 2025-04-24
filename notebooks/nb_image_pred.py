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

# 如果使用 Apple MPS，遇到不支持的操作时回退到 CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def setup_device():
    """设置计算设备并进行优化配置"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()  # 使用 bfloat16 混合精度
        if torch.cuda.get_device_properties(0).major >= 8:  # Ampere 架构 GPU
            torch.backends.cuda.matmul.allow_tf32 = True  # 启用 TF32 加速
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


def show_anns(anns, borders=True):
    """可视化分割掩码"""
    if not anns:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # 创建透明背景图像 (RGBA)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0  # 设置 alpha 通道为 0 (完全透明)

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])  # 随机颜色，透明度 0.5
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def get_memory_usage():
    """获取当前 GPU 显存占用（MB），若无 GPU 返回 0"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0, 0


def process_image(image_path, mask_generator, output_dir, logger):
    """处理单张图片并保存结果，记录处理时间和显存占用"""
    logger.info(f"正在处理图片: {image_path}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 记录推理前显存占用
    mem_before_alloc, mem_before_res = get_memory_usage()
    logger.info(f"推理前显存 - 分配: {mem_before_alloc:.2f} MB, 保留: {mem_before_res:.2f} MB")

    # 加载图像
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    # 生成掩码
    masks = mask_generator.generate(image)
    logger.info(f"生成掩码数量: {len(masks)}")

    # 记录推理后显存占用
    mem_after_alloc, mem_after_res = get_memory_usage()
    logger.info(f"推理后显存 - 分配: {mem_after_alloc:.2f} MB, 保留: {mem_after_res:.2f} MB")
    
    # 可视化并保存
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')

    # 保存到指定输出目录
    output_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_segmented.png")
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()  # 关闭当前图像，释放内存
    logger.info(f"结果已保存至: {output_filename}")

    # 清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("已清空缓存")

    # 计算处理时间
    process_time = time.time() - start_time
    logger.info(f"单图处理时间: {process_time:.2f} 秒")


def main():
    """主函数：处理命令行参数并执行批量图片分割，记录总推理时间"""
    # 设置随机种子，确保结果可复现
    np.random.seed(3)

    # 设置日志
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

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用 SAM2 对图片进行自动分割")
    parser.add_argument('--input_dir', type=str, default='/home/surgicalai/Data/images', help="输入图片目录路径")
    parser.add_argument('--output_dir', type=str, default="/home/surgicalai/Data/output/sam2", help="输出图片保存目录")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"输出目录: {args.output_dir}")

    # 设置设备
    device = setup_device()

    # 加载 SAM2 模型
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    logger.info("正在加载 SAM2 模型...")
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=8,
        points_per_batch=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.5,
        stability_score_offset=2.0,
        crop_n_layers=2,
        box_nms_thresh=2.0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=50,
        use_m2m=True,
    )
    logger.info("模型加载完成")

    # 获取输入目录中的所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(image_extensions)]
    if not image_files:
        logger.error(f"错误: 输入目录 {args.input_dir} 中未找到图片文件")
        sys.exit(1)

    logger.info(f"找到 {len(image_files)} 张图片，开始处理...")
    
    # 记录总推理开始时间
    total_start_time = time.time()
    
    # 循环处理每张图片
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(args.input_dir, image_file)
        logger.info(f"\n[{idx}/{len(image_files)}] 开始处理...")
        process_image(image_path, mask_generator, args.output_dir, logger)

    # 计算总推理时间
    total_time = time.time() - total_start_time
    logger.info(f"\n所有图片处理完成！总推理时间: {total_time:.2f} 秒")
    logger.info('-' * 10 + " Section ends here " + '-' * 12 + "\n")
    logger.info("\n")


if __name__ == "__main__":
    main()