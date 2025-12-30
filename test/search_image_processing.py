import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List
import itertools


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


# Interpolation modes for img2 resizing
INTERPOLATION_MODES = {
    "NEAREST": cv2.INTER_NEAREST,
    "LINEAR": cv2.INTER_LINEAR,
    "CUBIC": cv2.INTER_CUBIC,
    "AREA": cv2.INTER_AREA,
    "LANCZOS4": cv2.INTER_LANCZOS4,
}

# Smoothing parameters for img1
SMOOTHING_CONFIGS = [
    {"type": "none", "params": {}},
    {"type": "gaussian", "params": {"ksize": (3, 3), "sigmaX": 0.5}},
    {"type": "gaussian", "params": {"ksize": (3, 3), "sigmaX": 1.0}},
    {"type": "gaussian", "params": {"ksize": (5, 5), "sigmaX": 0.5}},
    {"type": "gaussian", "params": {"ksize": (5, 5), "sigmaX": 1.0}},
    {"type": "median", "params": {"ksize": 3}},
    {"type": "median", "params": {"ksize": 5}},
    {"type": "bilateral", "params": {"d": 5, "sigmaColor": 10, "sigmaSpace": 10}},
    {"type": "bilateral", "params": {"d": 5, "sigmaColor": 20, "sigmaSpace": 20}},
    {"type": "bilateral", "params": {"d": 9, "sigmaColor": 10, "sigmaSpace": 10}},
]


def apply_smoothing(img: np.ndarray, config: Dict) -> np.ndarray:
    """Apply smoothing/filtering to image based on config."""
    if config["type"] == "none":
        return img.copy()
    elif config["type"] == "gaussian":
        return cv2.GaussianBlur(img, **config["params"])
    elif config["type"] == "median":
        return cv2.medianBlur(img, config["params"]["ksize"])
    elif config["type"] == "bilateral":
        return cv2.bilateralFilter(img, **config["params"])
    else:
        raise ValueError(f"Unknown smoothing type: {config['type']}")


def resize_image(img: np.ndarray, target_shape: Tuple[int, int], interpolation: int) -> np.ndarray:
    """Resize image to target shape using specified interpolation."""
    return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=interpolation)


def compute_difference_metrics(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Compute various difference metrics between two images."""
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    diff = img1_float - img2_float
    abs_diff = np.abs(diff)
    
    metrics = {
        "mean_abs_diff": np.mean(abs_diff),
        "max_abs_diff": np.max(abs_diff),
        "rmse": np.sqrt(np.mean(diff ** 2)),
        "mean_diff": np.mean(diff),
        "std_diff": np.std(diff),
    }
    
    # Per-channel metrics if color image
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        metrics["mean_abs_diff_per_channel"] = np.mean(abs_diff, axis=(0, 1))
        metrics["rmse_per_channel"] = np.sqrt(np.mean(diff ** 2, axis=(0, 1)))
    
    return metrics


def search_best_processing(
    img1: np.ndarray,
    img2: np.ndarray,
    metric: str = "mean_abs_diff"
) -> Tuple[Dict, Dict, Dict]:
    """Search for best image processing combination.
    
    Returns:
        best_config: Dictionary with best smoothing config and interpolation mode
        best_metrics: Metrics for the best configuration
        all_results: List of all tested configurations and their metrics
    """
    all_results = []
    
    print(f"Searching over {len(SMOOTHING_CONFIGS)} smoothing configs and {len(INTERPOLATION_MODES)} interpolation modes...")
    print(f"Total combinations: {len(SMOOTHING_CONFIGS) * len(INTERPOLATION_MODES)}")
    print()
    
    for smooth_config in SMOOTHING_CONFIGS:
        for interp_name, interp_mode in INTERPOLATION_MODES.items():
            # Apply smoothing to img1
            img1_processed = apply_smoothing(img1, smooth_config)
            
            # Resize img2 with interpolation
            if img1.shape != img2.shape:
                img2_resized = resize_image(img2, (img1.shape[0], img1.shape[1]), interp_mode)
            else:
                img2_resized = img2.copy()
            
            # Compute metrics
            metrics = compute_difference_metrics(img1_processed, img2_resized)
            
            # Store result
            result = {
                "smoothing": smooth_config,
                "interpolation": interp_name,
                "metrics": metrics,
            }
            all_results.append(result)
            
            # Print progress for top candidates
            current_value = metrics[metric]
            if len(all_results) <= 5 or current_value < sorted([r["metrics"][metric] for r in all_results])[4]:
                print(f"  Smooth: {smooth_config['type']} {smooth_config['params']}, "
                      f"Interp: {interp_name}, {metric}: {current_value:.4f}")
    
    # Find best configuration
    best_result = min(all_results, key=lambda x: x["metrics"][metric])
    
    best_config = {
        "smoothing": best_result["smoothing"],
        "interpolation": best_result["interpolation"],
    }
    
    return best_config, best_result["metrics"], all_results


def print_results(best_config: Dict, best_metrics: Dict, all_results: List[Dict], metric: str):
    """Print search results."""
    print("\n" + "="*80)
    print("SEARCH RESULTS")
    print("="*80)
    
    print(f"\nBest configuration (minimizing {metric}):")
    print(f"  Smoothing: {best_config['smoothing']['type']} {best_config['smoothing']['params']}")
    print(f"  Interpolation: {best_config['interpolation']}")
    
    print(f"\nBest metrics:")
    print(f"  Mean absolute difference: {best_metrics['mean_abs_diff']:.6f}")
    print(f"  Max absolute difference: {best_metrics['max_abs_diff']:.6f}")
    print(f"  RMSE: {best_metrics['rmse']:.6f}")
    print(f"  Mean difference: {best_metrics['mean_diff']:.6f}")
    print(f"  Std difference: {best_metrics['std_diff']:.6f}")
    
    if "mean_abs_diff_per_channel" in best_metrics:
        print(f"  Mean absolute difference per channel (R, G, B): {best_metrics['mean_abs_diff_per_channel']}")
        print(f"  RMSE per channel (R, G, B): {best_metrics['rmse_per_channel']}")
    
    # Show top 5 configurations
    sorted_results = sorted(all_results, key=lambda x: x["metrics"][metric])
    print(f"\nTop 5 configurations (by {metric}):")
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"  {i}. Smooth: {result['smoothing']['type']} {result['smoothing']['params']}, "
              f"Interp: {result['interpolation']}, {metric}: {result['metrics'][metric]:.4f}")


def visualize_best(
    img1: np.ndarray,
    img2: np.ndarray,
    best_config: Dict,
    best_metrics: Dict,
    output_path: Path,
    name: str = "comparison"
):
    """Visualize the best processing configuration."""
    import matplotlib.pyplot as plt
    
    # Apply best processing
    img1_processed = apply_smoothing(img1, best_config["smoothing"])
    interp_mode = INTERPOLATION_MODES[best_config["interpolation"]]
    
    if img1.shape != img2.shape:
        img2_resized = resize_image(img2, (img1.shape[0], img1.shape[1]), interp_mode)
    else:
        img2_resized = img2.copy()
    
    # Compute difference
    img1_float = img1_processed.astype(np.float32)
    img2_float = img2_resized.astype(np.float32)
    diff = np.abs(img1_float - img2_float)
    
    # Normalize difference for visualization
    diff_normalized = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    
    # Create overlay
    overlay = (img1_float * 0.5 + img2_float * 0.5).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    axes[0, 0].imshow(img1_processed)
    smooth_str = f"{best_config['smoothing']['type']} {best_config['smoothing']['params']}"
    axes[0, 0].set_title(f"Image 1 (processed: {smooth_str})")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_resized)
    axes[0, 1].set_title(f"Image 2 (resized: {best_config['interpolation']})")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(f"Overlay (50% blend)\nMean abs diff: {best_metrics['mean_abs_diff']:.2f}")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_normalized)
    axes[1, 1].set_title(f"Absolute Difference\nRMSE: {best_metrics['rmse']:.2f}")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Search for best image processing to minimize difference between two video frames"
    )
    parser.add_argument("video1_path", type=str, help="Path to first video file (MP4)")
    parser.add_argument("video2_path", type=str, help="Path to second video file (MP4)")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for results")
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_abs_diff",
        choices=["mean_abs_diff", "rmse", "max_abs_diff"],
        help="Metric to minimize (default: mean_abs_diff)"
    )
    parser.add_argument("--name", type=str, default="comparison", help="Name for output files")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    video1_path = Path(args.video1_path)
    video2_path = Path(args.video2_path)
    output_dir = Path(args.output_dir)
    
    if not video1_path.exists():
        raise ValueError(f"Video 1 does not exist: {video1_path}")
    if not video2_path.exists():
        raise ValueError(f"Video 2 does not exist: {video2_path}")
    
    # Load first frames from videos
    print(f"Loading first frames from videos...")
    img1 = load_first_frame_from_video(video1_path)
    img2 = load_first_frame_from_video(video2_path)
    
    if img1 is None:
        raise ValueError(f"Failed to load first frame from video 1: {video1_path}")
    if img2 is None:
        raise ValueError(f"Failed to load first frame from video 2: {video2_path}")
    
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    print()
    
    # Search for best processing
    best_config, best_metrics, all_results = search_best_processing(img1, img2, args.metric)
    
    # Print results
    print_results(best_config, best_metrics, all_results, args.metric)
    
    # Visualize
    if not args.no_viz:
        viz_path = output_dir / f"{args.name}_best_processing.png"
        visualize_best(img1, img2, best_config, best_metrics, viz_path, args.name)
    
    print("\n" + "="*80)
    print("Search complete!")
    print("="*80)


if __name__ == "__main__":
    main()

