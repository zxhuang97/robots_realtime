import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
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
        # Load arm actions
        self.left_actions = np.load(os.path.join(self.episode_dir, "action-left-pos.npy"))
        self.right_actions = np.load(os.path.join(self.episode_dir, "action-right-pos.npy"))
        
        # # Load gripper actions
        # self.left_gripper_actions = np.load(os.path.join(self.episode_dir, "left-gripper_pos.npy"))
        # self.right_gripper_actions = np.load(os.path.join(self.episode_dir, "right-gripper_pos.npy"))
        
        # Load stored states (joint positions and gripper positions)
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
        
        # Initialize forward kinematics if enabled
        self.left_fk = None
        self.right_fk = None
        if self.compare_ee_space:
            if self.xml_path is None:
                # Default to YAM XML path
                workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.xml_path = os.path.join(workspace_root, "dependencies/i2rt/i2rt/robot_models/yam/yam.xml")
            self.xml_path = os.path.expanduser(self.xml_path)
            if os.path.exists(self.xml_path):
                self.left_fk = Kinematics(self.xml_path, self.site_name)
                self.right_fk = Kinematics(self.xml_path, self.site_name)
                print(f"Forward kinematics initialized with XML: {self.xml_path}, site: {self.site_name}")
            else:
                print(f"Warning: XML path {self.xml_path} not found. End effector comparison disabled.")
                self.compare_ee_space = False
        
        print(f"ReplayAgent initialized with {self.num_steps} steps from {self.episode_dir}")

    def get_initial_state(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get the initial state (first joint positions and gripper positions) from the trajectory.
        
        Returns:
            Dictionary mapping robot names ('left', 'right') to dicts with 'arm_pos' and 'gripper_pos' keys.
        """
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
                print("ReplayAgent: Looping back to start")
            else:
                self.current_step = self.num_steps - 1
                if not hasattr(self, '_comparison_done'):
                    replay_finished = True
        
        left_action = self.left_actions[self.current_step]
        right_action = self.right_actions[self.current_step]
        
        # Concatenate arm and gripper actions
        # Record observations if available
        if "left" in obs and "joint_pos" in obs["left"] and "gripper_pos" in obs["left"]:
            self.replayed_left_joint_pos.append(obs["left"]["joint_pos"].copy())
            self.replayed_left_gripper_pos.append(obs["left"]["gripper_pos"].copy())
        if "right" in obs and "joint_pos" in obs["right"] and "gripper_pos" in obs["right"]:
            self.replayed_right_joint_pos.append(obs["right"]["joint_pos"].copy())
            self.replayed_right_gripper_pos.append(obs["right"]["gripper_pos"].copy())
        
        self.current_step += 1
        
        # Compare trajectories if replay finished
        if replay_finished:
            self.compare_trajectories()
            self._comparison_done = True
        
        return {
            "left": {"pos": left_action},
            "right": {"pos": right_action},
        }

    def compare_trajectories(self) -> None:
        """Compare the replayed trajectory with the stored trajectory."""
        if len(self.replayed_left_joint_pos) == 0:
            print("Warning: No observations recorded during replay. Cannot compare trajectories.")
            return

        # Convert lists to numpy arrays
        replayed_left_joint = np.array(self.replayed_left_joint_pos)
        replayed_right_joint = np.array(self.replayed_right_joint_pos)
        replayed_left_gripper = np.array(self.replayed_left_gripper_pos)
        replayed_right_gripper = np.array(self.replayed_right_gripper_pos)

        # Ensure arrays are 2D
        if replayed_left_gripper.ndim == 1:
            replayed_left_gripper = replayed_left_gripper.reshape(-1, 1)
        if replayed_right_gripper.ndim == 1:
            replayed_right_gripper = replayed_right_gripper.reshape(-1, 1)

        # Align lengths (take minimum to avoid index errors)
        min_len = min(
            len(self.stored_left_joint_pos),
            len(self.stored_right_joint_pos),
            len(replayed_left_joint),
            len(replayed_right_joint)
        )

        stored_left_joint = self.stored_left_joint_pos[:min_len]
        stored_right_joint = self.stored_right_joint_pos[:min_len]
        stored_left_gripper = self.stored_left_gripper_pos[:min_len]
        stored_right_gripper = self.stored_right_gripper_pos[:min_len]
        replayed_left_joint = replayed_left_joint[:min_len]
        replayed_right_joint = replayed_right_joint[:min_len]
        replayed_left_gripper = replayed_left_gripper[:min_len]
        replayed_right_gripper = replayed_right_gripper[:min_len]

        # Compute joint space errors
        left_joint_error = np.abs(replayed_left_joint - stored_left_joint)
        right_joint_error = np.abs(replayed_right_joint - stored_right_joint)
        left_gripper_error = np.abs(replayed_left_gripper - stored_left_gripper.reshape(-1, 1))
        right_gripper_error = np.abs(replayed_right_gripper - stored_right_gripper.reshape(-1, 1))

        # Compute end effector space errors if enabled
        left_ee_pos_error = None
        left_ee_ori_error = None
        right_ee_pos_error = None
        right_ee_ori_error = None
        stored_left_ee_pos = None
        replayed_left_ee_pos = None
        stored_right_ee_pos = None
        replayed_right_ee_pos = None
        
        if self.compare_ee_space and self.left_fk is not None and self.right_fk is not None:
            # Compute end effector poses for stored trajectories
            stored_left_ee_poses = []
            stored_right_ee_poses = []
            for i in range(min_len):
                stored_left_ee_poses.append(self.left_fk.fk(stored_left_joint[i, :6]))
                stored_right_ee_poses.append(self.right_fk.fk(stored_right_joint[i, :6]))
            
            # Compute end effector poses for replayed trajectories
            replayed_left_ee_poses = []
            replayed_right_ee_poses = []
            for i in range(min_len):
                replayed_left_ee_poses.append(self.left_fk.fk(replayed_left_joint[i, :6]))
                replayed_right_ee_poses.append(self.right_fk.fk(replayed_right_joint[i, :6]))
            
            # Extract positions and orientations
            stored_left_ee_pos = np.array([pose[:3, 3] for pose in stored_left_ee_poses])
            stored_right_ee_pos = np.array([pose[:3, 3] for pose in stored_right_ee_poses])
            replayed_left_ee_pos = np.array([pose[:3, 3] for pose in replayed_left_ee_poses])
            replayed_right_ee_pos = np.array([pose[:3, 3] for pose in replayed_right_ee_poses])
            
            # Compute position errors
            left_ee_pos_error = np.linalg.norm(replayed_left_ee_pos - stored_left_ee_pos, axis=1)
            right_ee_pos_error = np.linalg.norm(replayed_right_ee_pos - stored_right_ee_pos, axis=1)
            
            # Compute orientation errors (using rotation angle)
            left_ee_ori_error = []
            right_ee_ori_error = []
            for i in range(min_len):
                R_stored_left = stored_left_ee_poses[i][:3, :3]
                R_replayed_left = replayed_left_ee_poses[i][:3, :3]
                R_diff_left = R_replayed_left @ R_stored_left.T
                angle_left = np.arccos(np.clip((np.trace(R_diff_left) - 1) / 2, -1, 1))
                left_ee_ori_error.append(angle_left)
                
                R_stored_right = stored_right_ee_poses[i][:3, :3]
                R_replayed_right = replayed_right_ee_poses[i][:3, :3]
                R_diff_right = R_replayed_right @ R_stored_right.T
                angle_right = np.arccos(np.clip((np.trace(R_diff_right) - 1) / 2, -1, 1))
                right_ee_ori_error.append(angle_right)
            
            left_ee_ori_error = np.array(left_ee_ori_error)
            right_ee_ori_error = np.array(right_ee_ori_error)

        # Print comparison results
        print("\n" + "="*60)
        print("TRAJECTORY COMPARISON RESULTS")
        print("="*60)
        print(f"Compared {min_len} steps")
        print(f"\nLeft Arm Joint Position Errors:")
        print(f"  Mean error: {np.mean(left_joint_error):.6f}")
        print(f"  Max error: {np.max(left_joint_error):.6f}")
        print(f"  RMS error: {np.sqrt(np.mean(left_joint_error**2)):.6f}")
        print(f"\nRight Arm Joint Position Errors:")
        print(f"  Mean error: {np.mean(right_joint_error):.6f}")
        print(f"  Max error: {np.max(right_joint_error):.6f}")
        print(f"  RMS error: {np.sqrt(np.mean(right_joint_error**2)):.6f}")
        
        if left_ee_pos_error is not None and left_ee_ori_error is not None and right_ee_pos_error is not None and right_ee_ori_error is not None:
            print(f"\nLeft Arm End Effector Position Errors:")
            print(f"  Mean error: {np.mean(left_ee_pos_error):.6f} m")
            print(f"  Max error: {np.max(left_ee_pos_error):.6f} m")
            print(f"  RMS error: {np.sqrt(np.mean(left_ee_pos_error**2)):.6f} m")
            print(f"\nLeft Arm End Effector Orientation Errors:")
            print(f"  Mean error: {np.mean(left_ee_ori_error):.6f} rad ({np.degrees(np.mean(left_ee_ori_error)):.4f} deg)")
            print(f"  Max error: {np.max(left_ee_ori_error):.6f} rad ({np.degrees(np.max(left_ee_ori_error)):.4f} deg)")
            print(f"  RMS error: {np.sqrt(np.mean(left_ee_ori_error**2)):.6f} rad ({np.degrees(np.sqrt(np.mean(left_ee_ori_error**2))):.4f} deg)")
            
            print(f"\nRight Arm End Effector Position Errors:")
            print(f"  Mean error: {np.mean(right_ee_pos_error):.6f} m")
            print(f"  Max error: {np.max(right_ee_pos_error):.6f} m")
            print(f"  RMS error: {np.sqrt(np.mean(right_ee_pos_error**2)):.6f} m")
            print(f"\nRight Arm End Effector Orientation Errors:")
            print(f"  Mean error: {np.mean(right_ee_ori_error):.6f} rad ({np.degrees(np.mean(right_ee_ori_error)):.4f} deg)")
            print(f"  Max error: {np.max(right_ee_ori_error):.6f} rad ({np.degrees(np.max(right_ee_ori_error)):.4f} deg)")
            print(f"  RMS error: {np.sqrt(np.mean(right_ee_ori_error**2)):.6f} rad ({np.degrees(np.sqrt(np.mean(right_ee_ori_error**2))):.4f} deg)")
        
        print(f"\nLeft Gripper Position Errors:")
        print(f"  Mean error: {np.mean(left_gripper_error):.6f}")
        print(f"  Max error: {np.max(left_gripper_error):.6f}")
        print(f"  RMS error: {np.sqrt(np.mean(left_gripper_error**2)):.6f}")
        print(f"\nRight Gripper Position Errors:")
        print(f"  Mean error: {np.mean(right_gripper_error):.6f}")
        print(f"  Max error: {np.max(right_gripper_error):.6f}")
        print(f"  RMS error: {np.sqrt(np.mean(right_gripper_error**2)):.6f}")
        print("="*60 + "\n")

        # Plot joint trajectories: one figure per arm, 2x4 grid (6 joints + 1 gripper = 7 subplots)
        timesteps = np.arange(min_len)

        # Left arm: 6 joints + 1 gripper in 2x4 grid
        num_left_joints = stored_left_joint.shape[1]
        fig_left, axes_left = plt.subplots(2, 4, sharex=True, figsize=(16, 8))
        fig_left.suptitle("Left Arm Trajectories (Stored vs Replayed)")
        axes_left_flat = axes_left.flatten()
        
        # Plot 6 joints
        for j in range(num_left_joints):
            ax = axes_left_flat[j]
            ax.plot(timesteps, stored_left_joint[:, j], label="Stored", linestyle="-")
            ax.plot(timesteps, replayed_left_joint[:, j], label="Replayed", linestyle="--")
            ax.set_ylabel(f"Joint {j}")
            ax.grid(True, linestyle=":")
            if j == 0:
                ax.legend()
            
            # Add error statistics text
            joint_error = left_joint_error[:, j]
            mean_err = np.mean(joint_error)
            max_err = np.max(joint_error)
            ax.text(0.98, 0.98, f"Mean: {mean_err:.6f}\nMax: {max_err:.6f}", 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot gripper (7th subplot)
        stored_left_gripper_flat = stored_left_gripper.flatten() if stored_left_gripper.ndim > 1 else stored_left_gripper
        replayed_left_gripper_flat = replayed_left_gripper.flatten() if replayed_left_gripper.ndim > 1 else replayed_left_gripper
        ax_gripper = axes_left_flat[6]
        ax_gripper.plot(timesteps, stored_left_gripper_flat[:min_len], label="Stored", linestyle="-")
        ax_gripper.plot(timesteps, replayed_left_gripper_flat[:min_len], label="Replayed", linestyle="--")
        ax_gripper.set_ylabel("Gripper")
        ax_gripper.grid(True, linestyle=":")
        
        # Add error statistics text for gripper
        gripper_error_flat = left_gripper_error.flatten()
        mean_err = np.mean(gripper_error_flat)
        max_err = np.max(gripper_error_flat)
        ax_gripper.text(0.98, 0.98, f"Mean: {mean_err:.6f}\nMax: {max_err:.6f}", 
                       transform=ax_gripper.transAxes, fontsize=9, verticalalignment='top',
                       horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide the 8th subplot (unused)
        axes_left_flat[7].axis('off')
        
        # Set xlabel on bottom row
        for j in range(4, 8):
            if j != 7:  # Skip the hidden subplot
                axes_left_flat[j].set_xlabel("Timestep")
        fig_left.tight_layout(rect=(0, 0.03, 1, 0.95))
        left_fig_path = os.path.join(self.episode_dir, "left_trajectory.png")
        fig_left.savefig(left_fig_path)

        # Right arm: 6 joints + 1 gripper in 2x4 grid
        num_right_joints = stored_right_joint.shape[1]
        fig_right, axes_right = plt.subplots(2, 4, sharex=True, figsize=(16, 8))
        fig_right.suptitle("Right Arm Trajectories (Stored vs Replayed)")
        axes_right_flat = axes_right.flatten()
        
        # Plot 6 joints
        for j in range(num_right_joints):
            ax = axes_right_flat[j]
            ax.plot(timesteps, stored_right_joint[:, j], label="Stored", linestyle="-")
            ax.plot(timesteps, replayed_right_joint[:, j], label="Replayed", linestyle="--")
            ax.set_ylabel(f"Joint {j}")
            ax.grid(True, linestyle=":")
            if j == 0:
                ax.legend()
            
            # Add error statistics text
            joint_error = right_joint_error[:, j]
            mean_err = np.mean(joint_error)
            max_err = np.max(joint_error)
            ax.text(0.98, 0.98, f"Mean: {mean_err:.6f}\nMax: {max_err:.6f}", 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot gripper (7th subplot)
        stored_right_gripper_flat = stored_right_gripper.flatten() if stored_right_gripper.ndim > 1 else stored_right_gripper
        replayed_right_gripper_flat = replayed_right_gripper.flatten() if replayed_right_gripper.ndim > 1 else replayed_right_gripper
        ax_gripper = axes_right_flat[6]
        ax_gripper.plot(timesteps, stored_right_gripper_flat[:min_len], label="Stored", linestyle="-")
        ax_gripper.plot(timesteps, replayed_right_gripper_flat[:min_len], label="Replayed", linestyle="--")
        ax_gripper.set_ylabel("Gripper")
        ax_gripper.grid(True, linestyle=":")
        
        # Add error statistics text for gripper
        gripper_error_flat = right_gripper_error.flatten()
        mean_err = np.mean(gripper_error_flat)
        max_err = np.max(gripper_error_flat)
        ax_gripper.text(0.98, 0.98, f"Mean: {mean_err:.6f}\nMax: {max_err:.6f}", 
                       transform=ax_gripper.transAxes, fontsize=9, verticalalignment='top',
                       horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide the 8th subplot (unused)
        axes_right_flat[7].axis('off')
        
        # Set xlabel on bottom row
        for j in range(4, 8):
            if j != 7:  # Skip the hidden subplot
                axes_right_flat[j].set_xlabel("Timestep")
        fig_right.tight_layout(rect=(0, 0.03, 1, 0.95))
        right_fig_path = os.path.join(self.episode_dir, "right_trajectory.png")
        fig_right.savefig(right_fig_path)

        plt.close(fig_left)
        plt.close(fig_right)
        
        # Plot end effector space comparison if enabled
        if (self.compare_ee_space and left_ee_pos_error is not None and left_ee_ori_error is not None 
            and right_ee_pos_error is not None and right_ee_ori_error is not None):
            self._plot_ee_trajectories(
                stored_left_ee_pos, replayed_left_ee_pos, left_ee_pos_error, left_ee_ori_error,
                stored_right_ee_pos, replayed_right_ee_pos, right_ee_pos_error, right_ee_ori_error,
                min_len
            )

    def _plot_ee_trajectories(
        self,
        stored_left_ee_pos: np.ndarray,
        replayed_left_ee_pos: np.ndarray,
        left_ee_pos_error: np.ndarray,
        left_ee_ori_error: np.ndarray,
        stored_right_ee_pos: np.ndarray,
        replayed_right_ee_pos: np.ndarray,
        right_ee_pos_error: np.ndarray,
        right_ee_ori_error: np.ndarray,
        min_len: int
    ) -> None:
        """Plot end effector trajectory comparisons."""
        timesteps = np.arange(min_len)
        
        # Left arm end effector plots: 3x2 grid (3 position axes + position error + orientation error + empty)
        fig_left_ee, axes_left_ee = plt.subplots(3, 2, sharex=True, figsize=(12, 12))
        fig_left_ee.suptitle("Left Arm End Effector Trajectories (Stored vs Replayed)")
        
        # Position plots (x, y, z)
        for i, axis_name in enumerate(['X', 'Y', 'Z']):
            ax = axes_left_ee[i, 0]
            ax.plot(timesteps, stored_left_ee_pos[:, i], label="Stored", linestyle="-")
            ax.plot(timesteps, replayed_left_ee_pos[:, i], label="Replayed", linestyle="--")
            ax.set_ylabel(f"Position {axis_name} (m)")
            ax.grid(True, linestyle=":")
            if i == 0:
                ax.legend()
            if i == 2:
                ax.set_xlabel("Timestep")
        
        # Position error plot
        ax = axes_left_ee[0, 1]
        ax.plot(timesteps, left_ee_pos_error, label="Position Error", color='red')
        ax.set_ylabel("Position Error (m)")
        ax.grid(True, linestyle=":")
        ax.legend()
        
        # Orientation error plot
        ax = axes_left_ee[1, 1]
        ax.plot(timesteps, np.degrees(left_ee_ori_error), label="Orientation Error", color='red')
        ax.set_ylabel("Orientation Error (deg)")
        ax.grid(True, linestyle=":")
        ax.legend()
        
        # Hide unused subplot
        axes_left_ee[2, 1].axis('off')
        axes_left_ee[2, 1].set_xlabel("Timestep")
        
        fig_left_ee.tight_layout(rect=(0, 0.03, 1, 0.95))
        left_ee_fig_path = os.path.join(self.episode_dir, "left_ee_trajectory.png")
        fig_left_ee.savefig(left_ee_fig_path)
        plt.close(fig_left_ee)
        
        # Right arm end effector plots: 3x2 grid
        fig_right_ee, axes_right_ee = plt.subplots(3, 2, sharex=True, figsize=(12, 12))
        fig_right_ee.suptitle("Right Arm End Effector Trajectories (Stored vs Replayed)")
        
        # Position plots (x, y, z)
        for i, axis_name in enumerate(['X', 'Y', 'Z']):
            ax = axes_right_ee[i, 0]
            ax.plot(timesteps, stored_right_ee_pos[:, i], label="Stored", linestyle="-")
            ax.plot(timesteps, replayed_right_ee_pos[:, i], label="Replayed", linestyle="--")
            ax.set_ylabel(f"Position {axis_name} (m)")
            ax.grid(True, linestyle=":")
            if i == 0:
                ax.legend()
            if i == 2:
                ax.set_xlabel("Timestep")
        
        # Position error plot
        ax = axes_right_ee[0, 1]
        ax.plot(timesteps, right_ee_pos_error, label="Position Error", color='red')
        ax.set_ylabel("Position Error (m)")
        ax.grid(True, linestyle=":")
        ax.legend()
        
        # Orientation error plot
        ax = axes_right_ee[1, 1]
        ax.plot(timesteps, np.degrees(right_ee_ori_error), label="Orientation Error", color='red')
        ax.set_ylabel("Orientation Error (deg)")
        ax.grid(True, linestyle=":")
        ax.legend()
        
        # Hide unused subplot
        axes_right_ee[2, 1].axis('off')
        axes_right_ee[2, 1].set_xlabel("Timestep")
        
        fig_right_ee.tight_layout(rect=(0, 0.03, 1, 0.95))
        right_ee_fig_path = os.path.join(self.episode_dir, "right_ee_trajectory.png")
        fig_right_ee.savefig(right_ee_fig_path)
        plt.close(fig_right_ee)

    @remote(serialization_needed=True)
    def action_spec(self) -> ActionSpec:
        """Define the action specification based on loaded data."""
        # Combined shape: arm joints + gripper (1)
        left_shape = (self.left_actions.shape[1],)
        right_shape = (self.right_actions.shape[1],)
        return {
            "left": {"pos": Array(shape=left_shape, dtype=self.left_actions.dtype)},
            "right": {"pos": Array(shape=right_shape, dtype=self.right_actions.dtype)},
        }

