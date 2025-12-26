import os
import time
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import imageio.v3 as iio

from robots_realtime.robots.robot import Robot
from robots_realtime.robots.utils import Rate
from robots_realtime.sensors.cameras.camera import CameraDriver
from robots_realtime.envs.robot_env import RobotEnv


class DatasetObservationEnv(RobotEnv):
    """Environment that returns observations from a dataset while still executing actions on hardware.
    
    This is useful for testing inference on dataset observations before running on real observations.
    Actions are executed on hardware, but if the predicted action differs significantly from the
    ground truth action in the dataset, the ground truth action is executed instead.
    
    This is different from ReplayAgent, which replays actions. This environment uses dataset
    observations for testing policy inference while still executing actions on the real robot.
    """
    
    def __init__(
        self,
        robot_dict: Dict[str, Robot],
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
        control_rate_hz: Union[Rate, float] = 100.0,
        use_joint_state_as_action: bool = False,
        reset_pos: Optional[np.ndarray] = None,
        home_pos: Optional[np.ndarray] = None,
        dataset_observation_dir: str = "",
        action_threshold: float = 0.05,
    ) -> None:
        super().__init__(
            robot_dict=robot_dict,
            camera_dict=camera_dict,
            control_rate_hz=control_rate_hz,
            use_joint_state_as_action=use_joint_state_as_action,
            reset_pos=reset_pos,
            home_pos=home_pos,
        )
        
        if not dataset_observation_dir:
            raise ValueError("dataset_observation_dir must be provided for DatasetObservationEnv")
        
        self._dataset_observation_dir = dataset_observation_dir
        self._action_threshold = action_threshold
        self._dataset_step = 0
        self._dataset_data = None
        
        self._load_dataset_data()
    
    def _load_dataset_data(self) -> None:
        """Load dataset observations and actions for inference testing."""
        if not os.path.exists(self._dataset_observation_dir):
            raise ValueError(f"Dataset directory not found: {self._dataset_observation_dir}")
        
        self._dataset_data = {
            "video_frames": {},
            "joint_pos": {},
            "gripper_pos": {},
            "gt_actions": {},
        }
        
        # Load robot states
        robot_names = list(self._robot_dict.keys())
        for robot_name in robot_names:
            # Try left/right naming convention
            prefix = "left" if robot_name == "left" else "right" if robot_name == "right" else robot_name
            
            joint_file = os.path.join(self._dataset_observation_dir, f"{prefix}-joint_pos.npy")
            gripper_file = os.path.join(self._dataset_observation_dir, f"{prefix}-gripper_pos.npy")
            action_file = os.path.join(self._dataset_observation_dir, f"action-{prefix}-pos.npy")
            
            if os.path.exists(joint_file):
                self._dataset_data["joint_pos"][robot_name] = np.load(joint_file)
            if os.path.exists(gripper_file):
                self._dataset_data["gripper_pos"][robot_name] = np.load(gripper_file)
            if os.path.exists(action_file):
                self._dataset_data["gt_actions"][robot_name] = np.load(action_file)
        
        # Load camera videos - preread all frames
        if self._camera_dict is not None:
            for camera_name in self._camera_dict.keys():
                video_path = os.path.join(self._dataset_observation_dir, f"{camera_name}-images-rgb.mp4")
                frames = iio.imread(video_path)
                self._dataset_data["video_frames"][camera_name] = np.array(frames)
        
        # Determine episode length
        lengths = []
        for robot_name in robot_names:
            if robot_name in self._dataset_data["joint_pos"]:
                lengths.append(len(self._dataset_data["joint_pos"][robot_name]))
            if robot_name in self._dataset_data["gt_actions"]:
                lengths.append(len(self._dataset_data["gt_actions"][robot_name]))
        self._dataset_length = min(lengths) if lengths else 0
        print(f"Loaded dataset with {self._dataset_length} steps from {self._dataset_observation_dir}")
    
    def get_obs(self) -> Dict[str, Any]:
        """Get observation from dataset at current step."""
        if self._dataset_data is None or self._dataset_step >= self._dataset_length:
            # Return real observation if dataset exhausted
            return super().get_obs()
        
        observations = {}
        observations["timestamp"] = time.time()
        
        # Get robot observations from dataset
        for robot_name in self._robot_dict.keys():
            obs = {}
            if robot_name in self._dataset_data["joint_pos"]:
                obs["joint_pos"] = self._dataset_data["joint_pos"][robot_name][self._dataset_step].copy()
            if robot_name in self._dataset_data["gripper_pos"]:
                obs["gripper_pos"] = self._dataset_data["gripper_pos"][robot_name][self._dataset_step].copy()
            observations[robot_name] = obs
        
        # Get camera observations from preread frames
        if self._camera_dict is not None:
            for camera_name in self._dataset_data["video_frames"].keys():
                frame_rgb = self._dataset_data["video_frames"][camera_name][self._dataset_step]
                observations[camera_name] = {"images": {"rgb": frame_rgb}}
        
        observations["timestamp_end"] = time.time()
        return observations
    
    def _compare_actions(self, predicted_action: Dict[str, Any], gt_action: Dict[str, Any]) -> bool:
        """Compare predicted action with GT action. Returns True if close enough."""
        if not predicted_action or not gt_action:
            return False
        
        # Compare actions for each robot
        for robot_name in predicted_action.keys():
            pred_val = predicted_action[robot_name]["pos"]
            gt_val = gt_action[robot_name]["pos"]
            diff = np.linalg.norm(pred_val - gt_val)
            if diff > self._action_threshold:
                print("-"*50)
                print("Current step: ", self._dataset_step)
                print("Predicted action: ", predicted_action[robot_name]["pos"])
                print("GT action: ", gt_action[robot_name]["pos"])
                print(f"Action difference too large for {robot_name}: {diff}")
                return False
        
        return True
    
    def _get_gt_action(self) -> Optional[Dict[str, Any]]:
        """Get ground truth action from dataset at current step."""        
        gt_action = {}
        for robot_name in self._robot_dict.keys():
            if robot_name in self._dataset_data["gt_actions"]:
                action = self._dataset_data["gt_actions"][robot_name][self._dataset_step].copy()
                gt_action[robot_name] = {"pos": action}
        
        return gt_action
    
    def step(self, action: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # type: ignore
        """Step the environment forward.

        Args:
            action: action to step the environment with.
            metadata: optional metadata dictionary.

        Returns:
            obs: observation from the environment.
        """
        # Compare predicted action with GT
        gt_action = self._get_gt_action()
        if self._compare_actions(action, gt_action):
            action_to_execute = action
        else:
            action_to_execute = gt_action
        ob = super().step(action_to_execute, metadata)
        self._dataset_step += 1
        self._dataset_step = min(self._dataset_step, self._dataset_length - 1)
        return ob
    
    def reset(self, reset_pos: Optional[Dict[str, Dict[str, np.ndarray]]] = None, duration: float = 2.0) -> Dict[str, Any]:  # type: ignore
        """Reset the environment and move to reset position if configured.
        
        Args:
            reset_pos: Optional reset position dictionary mapping robot names to dicts with 'arm_pos' and 'gripper_pos' keys.
                      If None, uses self._reset_pos if configured.
        """
        self._dataset_step = 0
        obs =  super().reset(reset_pos, duration)
        return obs
    
    def _get_obs_for_movement(self) -> Dict[str, Any]:
        """Get real observation for movement calculations.
        
        Overrides parent to return real robot observations instead of dataset observations,
        since we need to know the actual current robot position during movement.
        
        Returns:
            obs: Real observation from hardware
        """
        return super().get_obs()
    
    def close(self) -> None:
        """Close the environment."""
        super().close()

