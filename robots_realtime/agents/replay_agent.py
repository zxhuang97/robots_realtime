import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent
from robots_realtime.agents.constants import ActionSpec
from robots_realtime.utils.portal_utils import remote


@dataclass
class ReplayAgent(Agent):
    """An agent that replays actions from a dataset."""
    episode_dir: str
    loop: bool = False
    
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
        
        # Compute errors
        left_joint_error = np.abs(replayed_left_joint - stored_left_joint)
        right_joint_error = np.abs(replayed_right_joint - stored_right_joint)
        left_gripper_error = np.abs(replayed_left_gripper - stored_left_gripper.reshape(-1, 1))
        right_gripper_error = np.abs(replayed_right_gripper - stored_right_gripper.reshape(-1, 1))
        
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
        print(f"\nLeft Gripper Position Errors:")
        print(f"  Mean error: {np.mean(left_gripper_error):.6f}")
        print(f"  Max error: {np.max(left_gripper_error):.6f}")
        print(f"  RMS error: {np.sqrt(np.mean(left_gripper_error**2)):.6f}")
        print(f"\nRight Gripper Position Errors:")
        print(f"  Mean error: {np.mean(right_gripper_error):.6f}")
        print(f"  Max error: {np.max(right_gripper_error):.6f}")
        print(f"  RMS error: {np.sqrt(np.mean(right_gripper_error**2)):.6f}")
        print("="*60 + "\n")

    @remote(serialization_needed=True)
    def action_spec(self) -> ActionSpec:
        """Define the action specification based on loaded data."""
        # Combined shape: arm joints + gripper (1)
        left_shape = (self.left_arm_actions.shape[1] + 1,)
        right_shape = (self.right_arm_actions.shape[1] + 1,)
        return {
            "left": {"pos": Array(shape=left_shape, dtype=self.left_arm_actions.dtype)},
            "right": {"pos": Array(shape=right_shape, dtype=self.right_arm_actions.dtype)},
        }

