import argparse
import os
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Capture camera images and compare with dataset")
parser.add_argument("episode_folder", type=str, help="Path to episode folder containing video files")
args = parser.parse_args()

episode_folder = args.episode_folder
if not os.path.isdir(episode_folder):
    raise ValueError(f"Episode folder does not exist: {episode_folder}")

ctx = rs.context()
devices = ctx.query_devices()

print(f"Found {len(devices)} devices")
for i, dev in enumerate(devices):
    print(f"Device {i}:")
    print("  Name:", dev.get_info(rs.camera_info.name))
    print("  Serial:", dev.get_info(rs.camera_info.serial_number))

# Camera serial numbers
cameras = {
    "top": "230322274714",
    "left": "230322271210",
    "right": "230322277156"
}

pipelines = {}
configs = {}

# Create pipelines and configs for each camera
for name, serial in cameras.items():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipelines[name] = pipeline
    configs[name] = config

# Capture one image from each camera
captured_images = {}

for name, pipeline in pipelines.items():
    print(f"\nCapturing from {name} camera...")
    config = configs[name]
    
    # Start streaming
    pipeline.start(config)
    
    # Skip the first few frames to allow camera to stabilize
    print(f"  Waiting for {name} camera to stabilize...")
    for _ in range(60):  # Skip first 60 frames (~2 seconds at 30 fps)
        frames = pipeline.wait_for_frames()
    
    # Capture one frame
    print(f"  Capturing image from {name} camera...")
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if color_frame:
        color_image = np.asanyarray(color_frame.get_data())
        captured_images[name] = color_image
        print(f"  Successfully captured image from {name} camera")
    else:
        print(f"  Failed to capture image from {name} camera")
    
    # Stop streaming
    pipeline.stop()
    print(f"  Stopped {name} camera")

# Load first frame from dataset videos
print("\nLoading first frames from dataset videos...")
dataset_images = {}
video_files = {
    "top": "top_camera-images-rgb.mp4",
    "left": "left_camera-images-rgb.mp4",
    "right": "right_camera-images-rgb.mp4"
}

for name, video_file in video_files.items():
    video_path = os.path.join(episode_folder, video_file)
    if not os.path.exists(video_path):
        print(f"  Warning: {video_path} not found, skipping {name} camera")
        continue
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dataset_images[name] = frame_rgb
        print(f"  Loaded first frame from {name} camera video")
    else:
        print(f"  Failed to read first frame from {name} camera video")
    cap.release()

# Create comparison plots
print("\nCreating comparison plots...")
for name in ["top", "left", "right"]:
    if name not in captured_images or name not in dataset_images:
        print(f"  Skipping {name} camera (missing data)")
        continue
    
    dataset_img = dataset_images[name]
    new_img = captured_images[name]
    # Convert new image from BGR to RGB
    new_img_rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    
    # Create overlay (50% blend)
    overlay = (dataset_img.astype(np.float32) * 0.5 + new_img_rgb.astype(np.float32) * 0.5).astype(np.uint8)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(dataset_img)
    axes[0].set_title(f"{name.capitalize()} Camera - Dataset")
    axes[0].axis('off')
    
    axes[1].imshow(new_img_rgb)
    axes[1].set_title(f"{name.capitalize()} Camera - New Capture")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"{name.capitalize()} Camera - Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    output_path = f"{name}_camera_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved comparison plot to {output_path}")
    plt.close()

print("\nDone!")