import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import cv2
import pynvml
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import tempfile
import shutil
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Video segmentation with SAM 2 automatic mask generation")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input video frames (JPEG)")
parser.add_argument("--output_path", type=str, required=True, help="Base directory to save subvideo files (MP4)")
args = parser.parse_args()

video_dir = args.input_dir
output_base_dir = args.output_path

# Initialize pynvml for GPU memory monitoring
def init_pynvml():
    try:
        pynvml.nvmlInit()
        return True
    except pynvml.NVMLError:
        print("Failed to initialize pynvml. GPU memory monitoring disabled.")
        return False

# Get GPU memory usage
def get_gpu_memory(device_idx=0):
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1024**2, mem_info.total / 1024**2  # MB
    except pynvml.NVMLError:
        return None, None

# Select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Initialize pynvml if using CUDA
pynvml_initialized = init_pynvml() if device.type == "cuda" else False

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    used_mb, total_mb = get_gpu_memory()
    print(f"Initial GPU memory usage: {used_mb:.2f}/{total_mb:.2f} MB")
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS."
    )

# Load SAM 2 models
sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

print("Loading SAM 2 model for mask generation...")
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=64,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
)
print("SAM 2 mask generator loaded.")

print("Loading SAM 2 video predictor...")
from sam2.build_sam import build_sam2_video_predictor
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
print("SAM 2 video predictor loaded.")

# Function to create mask image
def create_mask_image(mask, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return (mask_image * 255).astype(np.uint8)

# Function to overlay mask on frame
def overlay_mask_on_frame(frame, mask_image):
    frame = np.array(frame)
    mask_image = mask_image[:, :, :3]
    overlay = cv2.addWeighted(frame, 0.7, mask_image, 0.3, 0)
    return overlay

# Scan all JPEG frame names
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
print(f"Found {len(frame_names)} frames in {video_dir}")

# Initialize video writer parameters
first_frame = Image.open(os.path.join(video_dir, frame_names[0]))
width, height = first_frame.size
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
FPS = 30.0
SUBVIDEO_FRAMES = int(30 * FPS)  # 30 seconds at 30 fps = 900 frames

# Create output directory
os.makedirs(output_base_dir, exist_ok=True)
print(f"Output directory created/verified: {output_base_dir}")

# Process video in blocks
BLOCK_SIZE = 100  # Number of frames per block
num_blocks = (len(frame_names) + BLOCK_SIZE - 1) // BLOCK_SIZE
print(f"Processing video in {num_blocks} blocks of {BLOCK_SIZE} frames each")

# Initialize video writer
video_writer = None
subvideo_idx = 0
global_obj_id = 1  # Incremental object ID across blocks

for block_idx in range(num_blocks):
    start_frame = block_idx * BLOCK_SIZE
    end_frame = min((block_idx + 1) * BLOCK_SIZE, len(frame_names))
    block_frames = frame_names[start_frame:end_frame]
    print(f"\nProcessing block {block_idx + 1}/{num_blocks} (frames {start_frame} to {end_frame - 1})")
    
    # Create temporary directory for block frames
    temp_dir = tempfile.mkdtemp()
    try:
        print(f"Copying {len(block_frames)} frames to temporary directory {temp_dir}...")
        for frame_idx, frame_name in enumerate(block_frames):
            src_path = os.path.join(video_dir, frame_name)
            dst_path = os.path.join(temp_dir, f"{frame_idx}.jpg")
            shutil.copy(src_path, dst_path)
        print("Frames copied.")
        
        # Clear cache before initializing state
        if device.type == "cuda":
            torch.cuda.empty_cache()
            used_mb, total_mb = get_gpu_memory()
            print(f"GPU memory before initializing state: {used_mb:.2f}/{total_mb:.2f} MB")
        
        # Initialize inference state with temporary directory
        print("Initializing inference state for block...")
        inference_state = predictor.init_state(video_path=temp_dir)
        predictor.reset_state(inference_state)
        print("Inference state initialized.")
        
        # Clear cache after initialization
        if device.type == "cuda":
            torch.cuda.empty_cache()
            used_mb, total_mb = get_gpu_memory()
            print(f"GPU memory after initializing state: {used_mb:.2f}/{total_mb:.2f} MB")
        
        # Generate automatic masks for the first frame of the block
        print("Generating automatic masks for first frame...")
        first_frame_path = os.path.join(temp_dir, "0.jpg")
        first_frame_img = np.array(Image.open(first_frame_path).convert("RGB"))
        masks = mask_generator.generate(first_frame_img)
        print(f"Generated {len(masks)} masks for first frame.")
        
        # Add masks as prompts for video predictor
        for mask_idx, mask_data in enumerate(masks[:5]):  # Limit to 5 masks
            obj_id = global_obj_id + mask_idx  # Unique object ID across blocks
            mask = mask_data["segmentation"].astype(np.uint8)
            print(f"Adding mask for object {obj_id} at frame 0")
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                mask=mask,
            )
            print(f"Processed mask for object {obj_id}")
        global_obj_id += len(masks[:5])  # Update global object ID
        
        # Clear cache after adding masks
        if device.type == "cuda":
            torch.cuda.empty_cache()
            used_mb, total_mb = get_gpu_memory()
            print(f"GPU memory after adding masks: {used_mb:.2f}/{total_mb:.2f} MB")
        
        # Propagate through block
        video_segments = {}
        print("Starting block propagation...")
        start_time = datetime.now()
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            global_frame_idx = start_frame + out_frame_idx
            if global_frame_idx >= end_frame:
                break
            video_segments[global_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            if out_frame_idx % 50 == 0:
                print(f"Processed frame {global_frame_idx}/{end_frame - 1} in block")
                if device.type == "cuda":
                    used_mb, total_mb = get_gpu_memory()
                    print(f"GPU memory at frame {global_frame_idx}: {used_mb:.2f}/{total_mb:.2f} MB")
        print(f"Block propagation completed in {datetime.now() - start_time}")
        
        # Clear cache after propagation
        if device.type == "cuda":
            torch.cuda.empty_cache()
            used_mb, total_mb = get_gpu_memory()
            print(f"GPU memory after propagation: {used_mb:.2f}/{total_mb:.2f} MB")
        
        # Generate video for this block
        print("Generating video for block...")
        for global_frame_idx in range(start_frame, end_frame):
            # Initialize new video writer if needed
            if global_frame_idx % SUBVIDEO_FRAMES == 0:
                if video_writer is not None:
                    video_writer.release()
                    print(f"Subvideo {subvideo_idx} saved (frames {global_frame_idx - SUBVIDEO_FRAMES} to {global_frame_idx - 1})")
                subvideo_path = os.path.join(output_base_dir, f"subvideo_{subvideo_idx}.mp4")
                video_writer = cv2.VideoWriter(subvideo_path, fourcc, FPS, (width, height))
                subvideo_idx += 1
                print(f"Initialized new video writer for {subvideo_path}")
            
            frame = Image.open(os.path.join(video_dir, frame_names[global_frame_idx]))
            frame_np = np.array(frame)
            
            # Create combined mask
            combined_mask = np.zeros((height, width, 4), dtype=np.uint8)
            if global_frame_idx in video_segments:
                for out_obj_id, out_mask in video_segments[global_frame_idx].items():
                    mask_image = create_mask_image(out_mask, obj_id=out_obj_id % 10)  # Modulo for color consistency
                    combined_mask = np.maximum(combined_mask, mask_image)
            
            # Overlay mask
            overlay_frame = overlay_mask_on_frame(frame_np, combined_mask)
            
            # Write to video
            overlay_frame_bgr = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(overlay_frame_bgr)
            
            if (global_frame_idx - start_frame) % 50 == 0:
                print(f"Written frame {global_frame_idx}/{end_frame - 1} to video")
        
        # Clear inference state and cache
        del inference_state
        if device.type == "cuda":
            torch.cuda.empty_cache()
            used_mb, total_mb = get_gpu_memory()
            print(f"GPU memory after block cleanup: {used_mb:.2f}/{total_mb:.2f} MB")
    
    finally:
        # Clean up temporary directory
        print(f"Removing temporary directory {temp_dir}...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Temporary directory removed.")

# Release final video writer
if video_writer is not None:
    video_writer.release()
    print(f"Final subvideo {subvideo_idx} saved (frames {max(0, len(frame_names) - SUBVIDEO_FRAMES)} to {len(frame_names) - 1})")

# Clean up pynvml
if pynvml_initialized:
    pynvml.nvmlShutdown()

# Final cache clear
if device.type == "cuda":
    torch.cuda.empty_cache()
    used_mb, total_mb = get_gpu_memory()
    print(f"Final GPU memory usage: {used_mb:.2f}/{total_mb:.2f} MB")