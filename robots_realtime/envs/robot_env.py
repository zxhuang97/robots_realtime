import time
from typing import Any, Dict, Optional, Union

import dm_env
import numpy as np

from robots_realtime.robots.robot import Robot
from robots_realtime.robots.utils import Rate
from robots_realtime.sensors.cameras.camera import CameraDriver
from robots_realtime.utils.portal_utils import return_futures

class RobotEnv(dm_env.Environment):
    # Abstract methods.
    """A environment with a dm_env.Environment interface for a robot arm setup."""

    def __init__(
        self,
        robot_dict: Dict[str, Robot],
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
        control_rate_hz: Union[Rate, float] = 100.0,
        use_joint_state_as_action: bool = False,
        reset_pos: Optional[np.ndarray] = None,
        home_pos: Optional[np.ndarray] = None,
    ) -> None:
        self._robot_dict = robot_dict
        if isinstance(control_rate_hz, Rate):
            self._rate = control_rate_hz
        else:
            self._rate = Rate(control_rate_hz)

        self._use_joint_state_as_action = use_joint_state_as_action
        # get camera dict
        self._camera_dict = camera_dict
        
        # Convert reset_pos and home_pos arrays to dictionaries
        # Get current robot observations to extract gripper positions
        robot_obs_dict = {}
        for robot_name, robot in robot_dict.items():
            robot_obs_dict[robot_name] = robot.get_observations()
        
        self._reset_pos: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self._home_pos: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        
        if reset_pos is not None:
            self._reset_pos = {}
            for robot_name in robot_dict.keys():
                self._reset_pos[robot_name] = {
                    "arm_pos": reset_pos,
                    "gripper_pos": robot_obs_dict[robot_name]["gripper_pos"],
                }
        
        if home_pos is not None:
            self._home_pos = {}
            for robot_name in robot_dict.keys():
                current_gripper_pos = robot_obs_dict[robot_name]["gripper_pos"]
                self._home_pos[robot_name] = {
                    "arm_pos": home_pos,
                    "gripper_pos": current_gripper_pos,
                }

    def robot(self, name: str) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot_dict[name]

    def get_all_robots(self) -> Dict[str, Robot]:
        return self._robot_dict

    def __len__(self) -> int:
        return 0

    def _apply_action(self, action_dict: Dict[str, Any]) -> None:
        with return_futures(*self._robot_dict.values()):  # type: ignore
            for name, action in action_dict.items():
                if name == "base":
                    self._robot_dict[name].command_target_vel(action)
                elif self._use_joint_state_as_action:
                    self._robot_dict[name].command_joint_state(action)
                else:
                    self._robot_dict[name].command_joint_pos(action["pos"])

    def step(self, action: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # type: ignore
        """Step the environment forward.

        Args:
            action: action to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        if len(action) != 0:
            # get action at time t
            self._apply_action(action)
        self._rate.sleep()  # sleep until next timestep
        # return observation at time t+1
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        observations["timestamp"] = time.time()

        assert self._camera_dict is not None, "Camera dictionary is not set."
        clients = list(self._camera_dict.values()) + list(self._robot_dict.values())

        camera_futures = {}
        robot_futures = {}
        with return_futures(*clients):  # type: ignore
            for name, client in self._camera_dict.items():
                camera_data = client.read()
                camera_futures[name] = camera_data
            for name, robot in self._robot_dict.items():
                robot_obs = robot.get_observations()
                robot_futures[name] = robot_obs

        for name, robot_obs_future in robot_futures.items():
            # start_time = time.time()
            robot_obs = robot_obs_future.result()
            observations[name] = robot_obs
            # end_time = time.time()
            # print(f"time taken to get robot data for {name}: {(end_time - start_time) * 1000} ms")

        for name, camera_data_future in camera_futures.items():
            # start_time = time.time()
            camera_data = camera_data_future.result()
            assert name not in observations
            observations[name] = camera_data
            # end_time = time.time()
            # print(f"time taken to get camera data for {name}: {(end_time - start_time) * 1000} ms")
        observations["timestamp_end"] = time.time()

        return observations

    def reset(self, reset_pos: Optional[Dict[str, Dict[str, np.ndarray]]] = None, duration: float = 2.0) -> Dict[str, Any]:  # type: ignore
        """Reset the environment and move to reset position if configured.
        
        Args:
            reset_pos: Optional reset position dictionary mapping robot names to dicts with 'arm_pos' and 'gripper_pos' keys.
                      If None, uses self._reset_pos if configured.
        """
        target_pos = reset_pos if reset_pos is not None else self._reset_pos
        if target_pos is not None:
            return self.move_to_target_slowly(target_pos, duration=duration)
        return self.get_obs()

    def _get_obs_for_movement(self) -> Dict[str, Any]:
        """Get observation to use during movement calculations.
        
        Can be overridden by subclasses to return real observations even when
        get_obs() is overridden to return dataset observations.
        
        Returns:
            obs: Observation to use for movement calculations
        """
        return self.get_obs()
    
    def move_to_target_slowly(self, target_pos: Optional[Dict[str, Dict[str, np.ndarray]]] = None, duration: float = 2.0) -> Dict[str, Any]:
        """Slowly move all robots to target position.
        
        Args:
            target_pos: Target joint positions dictionary mapping robot names to dicts with 'arm_pos' and 'gripper_pos' keys.
                       If None, robots will stay at current position.
            duration: Duration of the movement in seconds
            
        Returns:
            obs: Observation after reaching target position
        """
        obs = self._get_obs_for_movement()
        
        if target_pos is None:
            return self.get_obs()
        
        control_rate_hz = self._rate.rate
        assert control_rate_hz is not None, "Control rate must be set"
        num_steps = int(control_rate_hz * duration)
        robot_names = list(self._robot_dict.keys())
        for i in range(num_steps):
            alpha = i / num_steps if num_steps > 0 else 1.0
            action = {}
            for robot_name in robot_names:
                current_arm_pos = obs[robot_name]["joint_pos"]
                current_gripper_pos = obs[robot_name]["gripper_pos"]
                
                if robot_name in target_pos:
                    robot_target = target_pos[robot_name]
                    target_arm_pos = robot_target["arm_pos"]
                    target_gripper_pos = robot_target["gripper_pos"]
                else:
                    # If robot not in dict, keep current position
                    target_arm_pos = current_arm_pos
                    target_gripper_pos = current_gripper_pos
                
                command_arm_pos = target_arm_pos * alpha + current_arm_pos * (1 - alpha)
                command_gripper_pos = target_gripper_pos * alpha + current_gripper_pos * (1 - alpha)
                command_joint_pos = np.concatenate([command_arm_pos, command_gripper_pos])
                action[robot_name] = {"pos": command_joint_pos}
            self._apply_action(action)
            self._rate.sleep()
        return self.get_obs()

    def observation_spec(self):  # type: ignore
        return {}

    def action_spec(self):  # type: ignore
        spec = {}
        for name, robot in self._robot_dict.items():
            # if robot.get_robot_type() == RobotType.MOBILE_BASE:
            #     spec[name] = robot.joint_state_spec()
            # else:
            spec[name] = (
                robot.joint_state_spec() if self._use_joint_state_as_action else {"pos": robot.joint_pos_spec()}
            )
        return spec

    def close(self) -> None:
        """Close the environment, moving to reset_pos then home_pos if configured."""
        print("Closing environment...")
       
        if self._reset_pos is not None:
            print("Moving to reset position...")
            self.move_to_target_slowly(self._reset_pos, duration=2.0)
            print("Reached reset position")

        if self._home_pos is not None:
            print("Moving to home position...")
            self.move_to_target_slowly(self._home_pos, duration=2.0)
            print("Reached home position")

        
        # Close robots first to ensure safe shutdown
        for robot_name, robot in self._robot_dict.items():
            print(f"Closing robot {robot_name}")
            if hasattr(robot, "close"):
                robot.close()
        
        # Close cameras
        if self._camera_dict is not None:
            for camera_name, client in self._camera_dict.items():
                print(f"Closing camera {camera_name}")
                client.close()  # type: ignore

        print("Environment closed.")
