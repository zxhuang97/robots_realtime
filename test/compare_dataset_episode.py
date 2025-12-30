import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


def load_first_frame_from_video(video_path: Path) -> np.ndarray | None:
    """Load the first frame from a video file."""
    if not video_path.exists():
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB for consistency
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    return None


def load_episode_data(episode_dir: Path) -> dict:
    """Load action and joint position data from an episode directory."""
    
    data = {
        "left_actions": None,
        "right_actions": None,
        "left_joints": None,
        "right_joints": None,
        "video_frames": {},
    }
    
    # Load action files
    left_action_file = episode_dir / "action-left-pos.npy"
    right_action_file = episode_dir / "action-right-pos.npy"
    
    if left_action_file.exists():
        data["left_actions"] = np.load(left_action_file)
        print(f"  Loaded left actions: shape {data['left_actions'].shape}")
    else:
        print(f"  Warning: {left_action_file} not found")
    
    if right_action_file.exists():
        data["right_actions"] = np.load(right_action_file)
        print(f"  Loaded right actions: shape {data['right_actions'].shape}")
    else:
        print(f"  Warning: {right_action_file} not found")
    
    # Load joint position files
    left_joint_file = episode_dir / "left-joint_pos.npy"
    right_joint_file = episode_dir / "right-joint_pos.npy"
    
    if left_joint_file.exists():
        data["left_joints"] = np.load(left_joint_file)
        print(f"  Loaded left joints: shape {data['left_joints'].shape}")
    else:
        print(f"  Warning: {left_joint_file} not found")
    
    if right_joint_file.exists():
        data["right_joints"] = np.load(right_joint_file)
        print(f"  Loaded right joints: shape {data['right_joints'].shape}")
    else:
        print(f"  Warning: {right_joint_file} not found")
    
    # Load first frames from MP4 videos
    video_patterns = ["*_camera-images-rgb.mp4", "*_images-rgb.mp4", "*-images-rgb.mp4"]
    for pattern in video_patterns:
        for video_file in episode_dir.glob(pattern):
            camera_name = video_file.stem.replace("-images-rgb", "").replace("_images-rgb", "")
            frame = load_first_frame_from_video(video_file)
            if frame is not None:
                data["video_frames"][camera_name] = frame
                print(f"  Loaded first frame from {video_file.name}: shape {frame.shape}")
    
    return data


def compute_statistics(data1: np.ndarray, data2: np.ndarray, name: str) -> dict:
    """Compute comparison statistics between two arrays."""
    if data1 is None or data2 is None:
        return None
    
    # Handle different lengths
    min_len = min(len(data1), len(data2))
    data1_trunc = data1[:min_len]
    data2_trunc = data2[:min_len]
    
    diff = data1_trunc - data2_trunc
    abs_diff = np.abs(diff)
    
    stats = {
        "name": name,
        "length_1": len(data1),
        "length_2": len(data2),
        "min_length": min_len,
        "mean_abs_diff": np.mean(abs_diff),
        "max_abs_diff": np.max(abs_diff),
        "rmse": np.sqrt(np.mean(diff ** 2)),
        "mean_diff": np.mean(diff),
        "std_diff": np.std(diff),
    }
    
    # Per-dimension statistics if applicable
    if len(data1_trunc.shape) > 1:
        stats["mean_abs_diff_per_dim"] = np.mean(abs_diff, axis=0)
        stats["max_abs_diff_per_dim"] = np.max(abs_diff, axis=0)
        stats["rmse_per_dim"] = np.sqrt(np.mean(diff ** 2, axis=0))
    
    return stats


def plot_comparison(data1: np.ndarray, data2: np.ndarray, name: str, output_dir: Path):
    """Plot comparison between two arrays."""
    if data1 is None or data2 is None:
        return
    
    min_len = min(len(data1), len(data2))
    data1_trunc = data1[:min_len]
    data2_trunc = data2[:min_len]
    
    # Determine number of dimensions
    if len(data1_trunc.shape) == 1:
        n_dims = 1
    else:
        n_dims = data1_trunc.shape[1]
    
    # Create subplots
    if n_dims == 1:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes = [axes[0], axes[1]]
    else:
        n_cols = min(3, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=(15, 4 * (n_rows + 1)))
        axes = axes.flatten()
    
    time_steps = np.arange(min_len)
    
    # Plot each dimension
    for dim in range(n_dims):
        if n_dims == 1:
            dim_data1 = data1_trunc
            dim_data2 = data2_trunc
        else:
            dim_data1 = data1_trunc[:, dim]
            dim_data2 = data2_trunc[:, dim]
        
        if n_dims > 1:
            ax = axes[dim]
        else:
            ax = axes[0]
        
        ax.plot(time_steps, dim_data1, label="Episode 1", alpha=0.7, linewidth=1.5)
        ax.plot(time_steps, dim_data2, label="Episode 2", alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Time Step")
        ax.set_ylabel(f"Value (dim {dim})")
        ax.set_title(f"{name} - Dimension {dim}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot difference
    if n_dims == 1:
        diff_ax = axes[1]
        diff = data1_trunc - data2_trunc
        diff_ax.plot(time_steps, diff, label="Difference (E1 - E2)", color="red", alpha=0.7)
        diff_ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        diff_ax.set_xlabel("Time Step")
        diff_ax.set_ylabel("Difference")
        diff_ax.set_title(f"{name} - Difference")
        diff_ax.legend()
        diff_ax.grid(True, alpha=0.3)
    else:
        # Plot mean absolute difference across all dimensions
        diff_ax = axes[-1]
        diff = np.abs(data1_trunc - data2_trunc)
        mean_abs_diff = np.mean(diff, axis=1)
        diff_ax.plot(time_steps, mean_abs_diff, label="Mean Absolute Difference", color="red", alpha=0.7)
        diff_ax.set_xlabel("Time Step")
        diff_ax.set_ylabel("Mean |Difference|")
        diff_ax.set_title(f"{name} - Mean Absolute Difference")
        diff_ax.legend()
        diff_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    output_path = output_dir / f"{safe_name}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {output_path}")
    plt.close()


def compute_image_statistics(img1: np.ndarray, img2: np.ndarray, name: str) -> dict:
    """Compute comparison statistics between two images."""
    if img1 is None or img2 is None:
        return None
    
    # Ensure images have the same shape by resizing img2 to match img1
    if img1.shape != img2.shape:
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        img2_resized = img2
    
    # Convert to float for computation
    img1_float = img1.astype(np.float32)
    img2_float = img2_resized.astype(np.float32)
    
    diff = img1_float - img2_float
    abs_diff = np.abs(diff)
    
    stats = {
        "name": name,
        "shape_1": img1.shape,
        "shape_2": img2.shape,
        "mean_abs_diff": np.mean(abs_diff),
        "max_abs_diff": np.max(abs_diff),
        "rmse": np.sqrt(np.mean(diff ** 2)),
        "mean_diff": np.mean(diff),
        "std_diff": np.std(diff),
    }
    
    # Per-channel statistics if color image
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        stats["mean_abs_diff_per_channel"] = np.mean(abs_diff, axis=(0, 1))
        stats["max_abs_diff_per_channel"] = np.max(abs_diff, axis=(0, 1))
        stats["rmse_per_channel"] = np.sqrt(np.mean(diff ** 2, axis=(0, 1)))
    
    return stats


def plot_image_comparison(img1: np.ndarray, img2: np.ndarray, name: str, output_dir: Path):
    """Plot comparison between two images."""
    if img1 is None or img2 is None:
        return
    # Resize img2 to match img1
    if img1.shape != img2.shape:
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        img2_resized = img2
    
    # Compute difference
    img1_float = img1.astype(np.float32)
    img2_float = img2_resized.astype(np.float32)
    diff = np.abs(img1_float - img2_float)
    
    # Normalize difference for visualization (scale to 0-255)
    diff_normalized = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    
    # Create overlay (50% blend)
    overlay = (img1_float * 0.5 + img2_float * 0.5).astype(np.uint8)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title(f"{name} - Episode 1")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_resized)
    axes[0, 1].set_title(f"{name} - Episode 2 (resized)")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(f"{name} - Overlay (50% blend)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_normalized)
    axes[1, 1].set_title(f"{name} - Absolute Difference")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    output_path = output_dir / f"{safe_name}_first_frame_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {output_path}")
    plt.close()


def print_statistics(stats: dict):
    """Print comparison statistics."""
    if stats is None:
        return
    
    print(f"\n{stats['name']} Statistics:")
    if "length_1" in stats:
        print(f"  Episode 1 length: {stats['length_1']}")
        print(f"  Episode 2 length: {stats['length_2']}")
        print(f"  Compared length: {stats['min_length']}")
    else:
        print(f"  Episode 1 shape: {stats['shape_1']}")
        print(f"  Episode 2 shape: {stats['shape_2']}")
    print(f"  Mean absolute difference: {stats['mean_abs_diff']:.6f}")
    print(f"  Max absolute difference: {stats['max_abs_diff']:.6f}")
    print(f"  RMSE: {stats['rmse']:.6f}")
    print(f"  Mean difference: {stats['mean_diff']:.6f}")
    print(f"  Std difference: {stats['std_diff']:.6f}")
    
    if "mean_abs_diff_per_dim" in stats:
        print(f"  Mean absolute difference per dimension: {stats['mean_abs_diff_per_dim']}")
        print(f"  Max absolute difference per dimension: {stats['max_abs_diff_per_dim']}")
        print(f"  RMSE per dimension: {stats['rmse_per_dim']}")
    
    if "mean_abs_diff_per_channel" in stats:
        print(f"  Mean absolute difference per channel (R, G, B): {stats['mean_abs_diff_per_channel']}")
        print(f"  Max absolute difference per channel (R, G, B): {stats['max_abs_diff_per_channel']}")
        print(f"  RMSE per channel (R, G, B): {stats['rmse_per_channel']}")


def main():
    parser = argparse.ArgumentParser(description="Compare actions and joint angles between two episodes")
    parser.add_argument("episode1_dir", type=str, help="Path to first episode directory")
    parser.add_argument("episode2_dir", type=str, help="Path to second episode directory")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for plots (default: current directory)")
    args = parser.parse_args()
    
    episode1_dir = Path(args.episode1_dir)
    episode2_dir = Path(args.episode2_dir)
    output_dir = Path(args.output_dir)
    
    if not episode1_dir.exists():
        raise ValueError(f"Episode 1 directory does not exist: {episode1_dir}")
    if not episode2_dir.exists():
        raise ValueError(f"Episode 2 directory does not exist: {episode2_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Episode 1 from: {episode1_dir}")
    data1 = load_episode_data(episode1_dir)
    
    print(f"\nLoading Episode 2 from: {episode2_dir}")
    data2 = load_episode_data(episode2_dir)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # Compare left actions
    if data1["left_actions"] is not None and data2["left_actions"] is not None:
        stats = compute_statistics(data1["left_actions"], data2["left_actions"], "Left Actions")
        print_statistics(stats)
        plot_comparison(data1["left_actions"], data2["left_actions"], "Left Actions", output_dir)
    
    # Compare right actions
    if data1["right_actions"] is not None and data2["right_actions"] is not None:
        stats = compute_statistics(data1["right_actions"], data2["right_actions"], "Right Actions")
        print_statistics(stats)
        plot_comparison(data1["right_actions"], data2["right_actions"], "Right Actions", output_dir)
    
    # Compare left joints
    if data1["left_joints"] is not None and data2["left_joints"] is not None:
        stats = compute_statistics(data1["left_joints"], data2["left_joints"], "Left Joint Positions")
        print_statistics(stats)
        plot_comparison(data1["left_joints"], data2["left_joints"], "Left Joint Positions", output_dir)
    
    # Compare right joints
    if data1["right_joints"] is not None and data2["right_joints"] is not None:
        stats = compute_statistics(data1["right_joints"], data2["right_joints"], "Right Joint Positions")
        print_statistics(stats)
        plot_comparison(data1["right_joints"], data2["right_joints"], "Right Joint Positions", output_dir)
    
    # Compare video frames
    print("\n" + "-"*60)
    print("VIDEO FRAME COMPARISONS")
    print("-"*60)
    
    # Find common cameras
    common_cameras = set(data1["video_frames"].keys()) & set(data2["video_frames"].keys())
    
    if common_cameras:
        for camera_name in sorted(common_cameras):
            img1 = data1["video_frames"][camera_name]
            img2 = data2["video_frames"][camera_name]
            
            stats = compute_image_statistics(img1, img2, f"{camera_name} First Frame")
            print_statistics(stats)
            plot_image_comparison(img1, img2, f"{camera_name} First Frame", output_dir)
    else:
        print("  No common video files found between episodes")
    
    # Report cameras only in one episode
    only_ep1 = set(data1["video_frames"].keys()) - set(data2["video_frames"].keys())
    only_ep2 = set(data2["video_frames"].keys()) - set(data1["video_frames"].keys())
    
    if only_ep1:
        print(f"\n  Cameras only in Episode 1: {sorted(only_ep1)}")
    if only_ep2:
        print(f"  Cameras only in Episode 2: {sorted(only_ep2)}")
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

