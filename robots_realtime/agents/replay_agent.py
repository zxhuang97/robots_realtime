import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from dm_env.specs import Array

from i2rt.robots.kinematics import Kinematics
from robots_realtime.agents.agent import Agent
from robots_realtime.agents.constants import ActionSpec
from robots_realtime.utils.portal_utils import remote


@dataclass
class ReplayAgent(Agent):
    """An agent that replays actions from a dataset."""
    episode_dir: str
    loop: bool = False
    xml_path: Optional[str] = None
    site_name: str = "grasp_site"
    compare_ee_space: bool = True
    
    def __post_init__(self):
        # Load actions and stored states
        self.left_actions = np.load(os.path.join(self.episode_dir, "action-left-pos.npy"))
        self.right_actions = np.load(os.path.join(self.episode_dir, "action-right-pos.npy"))
        self.stored_left_joint_pos = np.load(os.path.join(self.episode_dir, "left-joint_pos.npy"))
        self.stored_right_joint_pos = np.load(os.path.join(self.episode_dir, "right-joint_pos.npy"))
        self.stored_left_gripper_pos = np.load(os.path.join(self.episode_dir, "left-gripper_pos.npy"))
        self.stored_right_gripper_pos = np.load(os.path.join(self.episode_dir, "right-gripper_pos.npy"))
        
        self.num_steps = len(self.left_actions)
        self.current_step = 0
        
        # Storage for replayed observations
        self.replayed_left_joint_pos = []
        self.replayed_right_joint_pos = []
        self.replayed_left_gripper_pos = []
        self.replayed_right_gripper_pos = []
        self.replayed_camera_images = {
            "top_camera": [],
            "left_camera": [],
            "right_camera": []
        }
        
        # Initialize forward kinematics if enabled
        self._init_forward_kinematics()

    def _init_forward_kinematics(self) -> None:
        """Initialize forward kinematics for end effector comparison."""
        self.left_fk = None
        self.right_fk = None
        if not self.compare_ee_space:
            return
        
        if self.xml_path is None:
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.xml_path = os.path.join(workspace_root, "dependencies/i2rt/i2rt/robot_models/yam/yam.xml")
        
        self.xml_path = os.path.expanduser(self.xml_path)
        if os.path.exists(self.xml_path):
            self.left_fk = Kinematics(self.xml_path, self.site_name)
            self.right_fk = Kinematics(self.xml_path, self.site_name)
        else:
            self.compare_ee_space = False

    def get_initial_state(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get the initial state (first joint positions and gripper positions) from the trajectory."""
        initial_state = {}
        init_index = 0
        if len(self.stored_left_joint_pos) > 0:
            initial_state["left"] = {
                "arm_pos": self.left_actions[init_index, :6],
                "gripper_pos": self.left_actions[init_index, 6]
            }
        if len(self.stored_right_joint_pos) > 0:
            initial_state["right"] = {
                "arm_pos": self.right_actions[init_index, :6],
                "gripper_pos": self.right_actions[init_index, 6]
            }
        return initial_state

    def act(self, obs: Dict[str, Any]) -> Any:
        """Returns the next action from the dataset and records observations."""
        replay_finished = False
        if self.current_step >= self.num_steps:
            if self.loop:
                self.current_step = 0
            else:
                self.current_step = self.num_steps - 1
                if not hasattr(self, '_comparison_done'):
                    replay_finished = True
        
        left_action = self.left_actions[self.current_step]
        right_action = self.right_actions[self.current_step]
        
        # Record observations
        self._record_observations(obs)
        self.current_step += 1
        
        # Compare trajectories if replay finished
        if replay_finished:
            self.compare_replay()
            self._comparison_done = True
        
        return {
            "left": {"pos": left_action},
            "right": {"pos": right_action},
        }

    def _record_observations(self, obs: Dict[str, Any]) -> None:
        """Record joint positions, gripper positions, and camera images from observations."""
        if "left" in obs and "joint_pos" in obs["left"] and "gripper_pos" in obs["left"]:
            self.replayed_left_joint_pos.append(obs["left"]["joint_pos"].copy())
            self.replayed_left_gripper_pos.append(obs["left"]["gripper_pos"].copy())
        if "right" in obs and "joint_pos" in obs["right"] and "gripper_pos" in obs["right"]:
            self.replayed_right_joint_pos.append(obs["right"]["joint_pos"].copy())
            self.replayed_right_gripper_pos.append(obs["right"]["gripper_pos"].copy())
        
        for camera_name in ["top_camera", "left_camera", "right_camera"]:
            if camera_name in obs and "images" in obs[camera_name] and "rgb" in obs[camera_name]["images"]:
                rgb_image = obs[camera_name]["images"]["rgb"]
                if rgb_image is not None:
                    self.replayed_camera_images[camera_name].append(rgb_image.copy())

    def compare_replay(self) -> None:
        """Compare both trajectories and camera images from the replay."""
        self.compare_trajectories()
        try:
            self.compare_camera_images()
        except Exception as e:
            print(f"Error during camera image comparison: {e}")
            import traceback
            traceback.print_exc()

    def compare_trajectories(self) -> None:
        """Compare the replayed trajectory with the stored trajectory."""
        if len(self.replayed_left_joint_pos) == 0:
            print("Warning: No observations recorded during replay. Cannot compare trajectories.")
            return

        # Prepare data
        data = self._prepare_trajectory_data()
        if data is None:
            return
        
        # Compute errors
        errors = self._compute_trajectory_errors(data)
        
        # Print statistics
        self._print_trajectory_statistics(errors, data["min_len"])
        
        # Plot trajectories
        self._plot_joint_trajectories(data, errors)
        if self.compare_ee_space and errors["ee_errors"] is not None:
            self._plot_ee_trajectories(errors["ee_errors"], data["min_len"])

    def _prepare_trajectory_data(self) -> Optional[Dict[str, Any]]:
        """Prepare and align trajectory data for comparison."""
        replayed_left_joint = np.array(self.replayed_left_joint_pos)
        replayed_right_joint = np.array(self.replayed_right_joint_pos)
        replayed_left_gripper = np.array(self.replayed_left_gripper_pos).reshape(-1, 1) if np.array(self.replayed_left_gripper_pos).ndim == 1 else np.array(self.replayed_left_gripper_pos)
        replayed_right_gripper = np.array(self.replayed_right_gripper_pos).reshape(-1, 1) if np.array(self.replayed_right_gripper_pos).ndim == 1 else np.array(self.replayed_right_gripper_pos)
        
        min_len = min(
            len(self.stored_left_joint_pos),
            len(self.stored_right_joint_pos),
            len(replayed_left_joint),
            len(replayed_right_joint)
        )
        
        return {
            "min_len": min_len,
            "stored_left_joint": self.stored_left_joint_pos[:min_len],
            "stored_right_joint": self.stored_right_joint_pos[:min_len],
            "stored_left_gripper": self.stored_left_gripper_pos[:min_len],
            "stored_right_gripper": self.stored_right_gripper_pos[:min_len],
            "replayed_left_joint": replayed_left_joint[:min_len],
            "replayed_right_joint": replayed_right_joint[:min_len],
            "replayed_left_gripper": replayed_left_gripper[:min_len],
            "replayed_right_gripper": replayed_right_gripper[:min_len],
        }

    def _compute_trajectory_errors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute joint space and end effector space errors."""
        # Joint space errors
        joint_errors = {
            "left": np.abs(data["replayed_left_joint"] - data["stored_left_joint"]),
            "right": np.abs(data["replayed_right_joint"] - data["stored_right_joint"]),
        }
        gripper_errors = {
            "left": np.abs(data["replayed_left_gripper"] - data["stored_left_gripper"].reshape(-1, 1)),
            "right": np.abs(data["replayed_right_gripper"] - data["stored_right_gripper"].reshape(-1, 1)),
        }
        
        # End effector errors
        ee_errors = None
        if self.compare_ee_space and self.left_fk is not None and self.right_fk is not None:
            ee_errors = self._compute_ee_errors(data)
        
        return {
            "joint_errors": joint_errors,
            "gripper_errors": gripper_errors,
            "ee_errors": ee_errors,
        }

    def _compute_ee_errors(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute end effector position and orientation errors."""
        if self.left_fk is None or self.right_fk is None:
            raise RuntimeError("Forward kinematics not initialized")
        
        min_len = data["min_len"]
        
        # Compute EE poses
        stored_left_ee_poses = [self.left_fk.fk(data["stored_left_joint"][i, :6]) for i in range(min_len)]
        stored_right_ee_poses = [self.right_fk.fk(data["stored_right_joint"][i, :6]) for i in range(min_len)]
        replayed_left_ee_poses = [self.left_fk.fk(data["replayed_left_joint"][i, :6]) for i in range(min_len)]
        replayed_right_ee_poses = [self.right_fk.fk(data["replayed_right_joint"][i, :6]) for i in range(min_len)]
        
        # Extract positions
        stored_left_ee_pos = np.array([pose[:3, 3] for pose in stored_left_ee_poses])
        stored_right_ee_pos = np.array([pose[:3, 3] for pose in stored_right_ee_poses])
        replayed_left_ee_pos = np.array([pose[:3, 3] for pose in replayed_left_ee_poses])
        replayed_right_ee_pos = np.array([pose[:3, 3] for pose in replayed_right_ee_poses])
        
        # Position errors
        left_ee_pos_error = np.linalg.norm(replayed_left_ee_pos - stored_left_ee_pos, axis=1)
        right_ee_pos_error = np.linalg.norm(replayed_right_ee_pos - stored_right_ee_pos, axis=1)
        
        # Orientation errors
        left_ee_ori_error = np.array([
            self._compute_rotation_angle(stored_left_ee_poses[i][:3, :3], replayed_left_ee_poses[i][:3, :3])
            for i in range(min_len)
        ])
        right_ee_ori_error = np.array([
            self._compute_rotation_angle(stored_right_ee_poses[i][:3, :3], replayed_right_ee_poses[i][:3, :3])
            for i in range(min_len)
        ])
        
        return {
            "stored_left_ee_pos": stored_left_ee_pos,
            "replayed_left_ee_pos": replayed_left_ee_pos,
            "left_ee_pos_error": left_ee_pos_error,
            "left_ee_ori_error": left_ee_ori_error,
            "stored_right_ee_pos": stored_right_ee_pos,
            "replayed_right_ee_pos": replayed_right_ee_pos,
            "right_ee_pos_error": right_ee_pos_error,
            "right_ee_ori_error": right_ee_ori_error,
        }

    def _compute_rotation_angle(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """Compute rotation angle between two rotation matrices."""
        R_diff = R2 @ R1.T
        return np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))

    def _print_trajectory_statistics(self, errors: Dict[str, Any], min_len: int) -> None:
        """Print trajectory comparison statistics."""
        print("\n" + "="*60)
        print("TRAJECTORY COMPARISON RESULTS")
        print("="*60)
        print(f"Compared {min_len} steps")
        
        for arm in ["left", "right"]:
            joint_err = errors["joint_errors"][arm]
            gripper_err = errors["gripper_errors"][arm]
            print(f"\n{arm.capitalize()} Arm Joint Position Errors:")
            print(f"  Mean error: {np.mean(joint_err):.6f}")
            print(f"  Max error: {np.max(joint_err):.6f}")
            print(f"\n{arm.capitalize()} Gripper Position Errors:")
            print(f"  Mean error: {np.mean(gripper_err):.6f}")
            print(f"  Max error: {np.max(gripper_err):.6f}")
        
        if errors["ee_errors"] is not None:
            ee = errors["ee_errors"]
            for arm in ["left", "right"]:
                pos_err = ee[f"{arm}_ee_pos_error"]
                ori_err = ee[f"{arm}_ee_ori_error"]
                print(f"\n{arm.capitalize()} Arm End Effector Position Errors:")
                print(f"  Mean error: {np.mean(pos_err):.6f} m")
                print(f"  Max error: {np.max(pos_err):.6f} m")
                print(f"\n{arm.capitalize()} Arm End Effector Orientation Errors:")
                print(f"  Mean error: {np.mean(ori_err):.6f} rad ({np.degrees(np.mean(ori_err)):.4f} deg)")
                print(f"  Max error: {np.max(ori_err):.6f} rad ({np.degrees(np.max(ori_err)):.4f} deg)")
        
        print("="*60 + "\n")

    def _plot_joint_trajectories(self, data: Dict[str, Any], errors: Dict[str, Any]) -> None:
        """Plot joint trajectories for both arms."""
        timesteps = np.arange(data["min_len"])
        
        for arm in ["left", "right"]:
            stored_joint = data[f"stored_{arm}_joint"]
            replayed_joint = data[f"replayed_{arm}_joint"]
            stored_gripper = data[f"stored_{arm}_gripper"]
            replayed_gripper = data[f"replayed_{arm}_gripper"]
            joint_error = errors["joint_errors"][arm]
            gripper_error = errors["gripper_errors"][arm]
            
            fig, axes = plt.subplots(2, 4, sharex=True, figsize=(16, 8))
            fig.suptitle(f"{arm.capitalize()} Arm Trajectories (Stored vs Replayed)")
            axes_flat = axes.flatten()
            
            # Plot joints
            num_joints = stored_joint.shape[1]
            for j in range(num_joints):
                ax = axes_flat[j]
                ax.plot(timesteps, stored_joint[:, j], label="Stored", linestyle="-")
                ax.plot(timesteps, replayed_joint[:, j], label="Replayed", linestyle="--")
                ax.set_ylabel(f"Joint {j}")
                ax.grid(True, linestyle=":")
                if j == 0:
                    ax.legend()
                self._add_error_text(ax, joint_error[:, j])
            
            # Plot gripper
            stored_gripper_flat = stored_gripper.flatten() if stored_gripper.ndim > 1 else stored_gripper
            replayed_gripper_flat = replayed_gripper.flatten() if replayed_gripper.ndim > 1 else replayed_gripper
            ax_gripper = axes_flat[6]
            ax_gripper.plot(timesteps, stored_gripper_flat, label="Stored", linestyle="-")
            ax_gripper.plot(timesteps, replayed_gripper_flat, label="Replayed", linestyle="--")
            ax_gripper.set_ylabel("Gripper")
            ax_gripper.grid(True, linestyle=":")
            self._add_error_text(ax_gripper, gripper_error.flatten())
            
            axes_flat[7].axis('off')
            for j in range(4, 8):
                if j != 7:
                    axes_flat[j].set_xlabel("Timestep")
            
            fig.tight_layout(rect=(0, 0.03, 1, 0.95))
            fig_path = os.path.join(self.episode_dir, f"{arm}_trajectory.png")
            fig.savefig(fig_path)
            plt.close(fig)

    def _plot_ee_trajectories(self, ee_errors: Dict[str, np.ndarray], min_len: int) -> None:
        """Plot end effector trajectory comparisons for both arms."""
        timesteps = np.arange(min_len)
        
        for arm in ["left", "right"]:
            stored_pos = ee_errors[f"stored_{arm}_ee_pos"]
            replayed_pos = ee_errors[f"replayed_{arm}_ee_pos"]
            pos_error = ee_errors[f"{arm}_ee_pos_error"]
            ori_error = ee_errors[f"{arm}_ee_ori_error"]
            
            fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 12))
            fig.suptitle(f"{arm.capitalize()} Arm End Effector Trajectories (Stored vs Replayed)")
            
            # Position plots
            for i, axis_name in enumerate(['X', 'Y', 'Z']):
                ax = axes[i, 0]
                ax.plot(timesteps, stored_pos[:, i], label="Stored", linestyle="-")
                ax.plot(timesteps, replayed_pos[:, i], label="Replayed", linestyle="--")
                ax.set_ylabel(f"Position {axis_name} (m)")
                ax.grid(True, linestyle=":")
                if i == 0:
                    ax.legend()
                if i == 2:
                    ax.set_xlabel("Timestep")
            
            # Position error
            ax = axes[0, 1]
            ax.plot(timesteps, pos_error, label="Position Error", color='red')
            ax.set_ylabel("Position Error (m)")
            ax.grid(True, linestyle=":")
            ax.legend()
            self._add_error_text(ax, pos_error, unit="m")
            
            # Orientation error
            ax = axes[1, 1]
            ax.plot(timesteps, np.degrees(ori_error), label="Orientation Error", color='red')
            ax.set_ylabel("Orientation Error (deg)")
            ax.grid(True, linestyle=":")
            ax.legend()
            self._add_error_text(ax, ori_error, unit="deg", convert_degrees=True)
            
            axes[2, 1].axis('off')
            axes[2, 1].set_xlabel("Timestep")
            
            fig.tight_layout(rect=(0, 0.03, 1, 0.95))
            fig_path = os.path.join(self.episode_dir, f"{arm}_ee_trajectory.png")
            fig.savefig(fig_path)
            plt.close(fig)

    def _add_error_text(self, ax: Any, error: np.ndarray, unit: str = "", convert_degrees: bool = False) -> None:
        """Add error statistics text to a plot axis."""
        mean_err = np.mean(error)
        max_err = np.max(error)
        if convert_degrees:
            mean_err_display = np.degrees(mean_err)
            max_err_display = np.degrees(max_err)
            text = f"Mean: {mean_err_display:.4f} deg\nMax: {max_err_display:.4f} deg"
        else:
            text = f"Mean: {mean_err:.6f}{unit}\nMax: {max_err:.6f}{unit}"
        ax.text(0.98, 0.98, text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def compare_camera_images(self) -> None:
        """Save replayed camera images as videos and compare with dataset videos."""
        total_images = sum(len(images) for images in self.replayed_camera_images.values())
        if total_images == 0:
            return
        
        print("\n" + "="*60)
        print("CAMERA IMAGE COMPARISON")
        print("="*60)
        
        video_files = {
            "top_camera": "top_camera-images-rgb.mp4",
            "left_camera": "left_camera-images-rgb.mp4",
            "right_camera": "right_camera-images-rgb.mp4"
        }
        
        for camera_name in ["top_camera", "left_camera", "right_camera"]:
            images = self.replayed_camera_images[camera_name]
            if len(images) == 0:
                continue
            
            dataset_video_path = os.path.join(self.episode_dir, video_files[camera_name])
            if not os.path.exists(dataset_video_path):
                continue
            
            self._create_side_by_side_video(camera_name, images, dataset_video_path)
            self._create_camera_comparison_plot(camera_name, images, dataset_video_path)
        
        print("="*60 + "\n")

    def _create_side_by_side_video(self, camera_name: str, images: list, dataset_video_path: str) -> None:
        """Create side-by-side comparison video."""
        fps = 30
        cap = cv2.VideoCapture(dataset_video_path)
        if not cap.isOpened():
            print(f"  Error: Failed to open dataset video {dataset_video_path}")
            return
        
        dataset_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dataset_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        replayed_height, replayed_width, _ = images[0].shape
        
        target_height = max(dataset_height, replayed_height)
        target_width_dataset = int(dataset_width * target_height / dataset_height)
        target_width_replayed = int(replayed_width * target_height / replayed_height)
        side_by_side_width = target_width_dataset + target_width_replayed
        
        side_by_side_path = os.path.join(self.episode_dir, f"{camera_name}_side_by_side_comparison.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(side_by_side_path, fourcc, fps, (side_by_side_width, target_height))
        
        if not writer.isOpened():
            cap.release()
            return
        
        max_frames = min(len(images), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        for frame_count in range(max_frames):
            ret, dataset_frame = cap.read()
            replayed_frame_bgr = cv2.cvtColor(images[frame_count], cv2.COLOR_RGB2BGR)
            
            dataset_resized = cv2.resize(dataset_frame, (target_width_dataset, target_height))
            replayed_resized = cv2.resize(replayed_frame_bgr, (target_width_replayed, target_height))
            side_by_side_frame = np.hstack([dataset_resized, replayed_resized])
            
            cv2.putText(side_by_side_frame, "Dataset", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(side_by_side_frame, "Replayed", (target_width_dataset + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            writer.write(side_by_side_frame)
        
        cap.release()
        writer.release()
        print(f"  Saved {camera_name} comparison video ({max_frames} frames)")

    def _create_camera_comparison_plot(self, camera_name: str, images: list, dataset_video_path: str) -> None:
        """Create comparison plot for first frame (dataset vs replayed with overlay)."""
        cap = cv2.VideoCapture(dataset_video_path)
        ret, dataset_frame = cap.read()
        cap.release()
        
        if not ret or len(images) == 0:
            return
        
        # Convert dataset frame from BGR to RGB
        dataset_img_rgb = cv2.cvtColor(dataset_frame, cv2.COLOR_BGR2RGB)
        replayed_img_rgb = images[0]
        
        # Resize if shapes don't match
        if dataset_img_rgb.shape != replayed_img_rgb.shape:
            dataset_img_rgb = cv2.resize(dataset_img_rgb, (replayed_img_rgb.shape[1], replayed_img_rgb.shape[0]))
        
        # Create color masks: blue for dataset, red for replayed
        dataset_float = dataset_img_rgb.astype(np.float32)
        replayed_float = replayed_img_rgb.astype(np.float32)
        
        # Add blue tint to dataset (boost blue channel, reduce red/green)
        dataset_tinted = dataset_float.copy()
        dataset_tinted[:, :, 2] = np.clip(dataset_tinted[:, :, 2] * 1.3, 0, 255)  # Boost blue
        dataset_tinted[:, :, 0] = np.clip(dataset_tinted[:, :, 0] * 0.8, 0, 255)  # Reduce red
        dataset_tinted[:, :, 1] = np.clip(dataset_tinted[:, :, 1] * 0.8, 0, 255)  # Reduce green
        
        # Add red tint to replayed (boost red channel, reduce green/blue)
        replayed_tinted = replayed_float.copy()
        replayed_tinted[:, :, 0] = np.clip(replayed_tinted[:, :, 0] * 1.3, 0, 255)  # Boost red
        replayed_tinted[:, :, 1] = np.clip(replayed_tinted[:, :, 1] * 0.8, 0, 255)  # Reduce green
        replayed_tinted[:, :, 2] = np.clip(replayed_tinted[:, :, 2] * 0.8, 0, 255)  # Reduce blue
        
        # Create overlay (50% blend with color tints)
        overlay = (dataset_tinted * 0.5 + replayed_tinted * 0.5).astype(np.uint8)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        camera_title = camera_name.replace('_', ' ').title()
        
        axes[0].imshow(dataset_img_rgb)
        axes[0].set_title(f"{camera_title} - Dataset")
        axes[0].axis('off')
        
        axes[1].imshow(replayed_img_rgb)
        axes[1].set_title(f"{camera_title} - Replayed")
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"{camera_title} - Overlay")
        axes[2].axis('off')
        
        # Add legend to overlay plot
        legend_elements = [
            Patch(facecolor='blue', alpha=0.6, label='Dataset'),
            Patch(facecolor='red', alpha=0.6, label='Replayed')
        ]
        axes[2].legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        plt.tight_layout()
        comparison_path = os.path.join(self.episode_dir, f"{camera_name}_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {camera_name} comparison plot")
        plt.close()


    @remote(serialization_needed=True)
    def action_spec(self) -> ActionSpec:
        """Define the action specification based on loaded data."""
        left_shape = (self.left_actions.shape[1],)
        right_shape = (self.right_actions.shape[1],)
        return {
            "left": {"pos": Array(shape=left_shape, dtype=self.left_actions.dtype)},
            "right": {"pos": Array(shape=right_shape, dtype=self.right_actions.dtype)},
        }
