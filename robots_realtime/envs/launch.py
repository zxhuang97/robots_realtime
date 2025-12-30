"""
Main launch script for YAM realtime robot control environment.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tyro

from robots_realtime.agents.agent import Agent
from robots_realtime.agents.replay_agent import ReplayAgent
from robots_realtime.envs.configs.instantiate import instantiate
from robots_realtime.envs.configs.loader import DictLoader
from robots_realtime.envs.robot_env import RobotEnv
from robots_realtime.envs.dataset_observation_env import DatasetObservationEnv
from robots_realtime.robots.robot import Robot
from robots_realtime.robots.utils import Rate, Timeout
from robots_realtime.sensors.cameras.camera import CameraDriver
from robots_realtime.utils.launch_utils import (
    cleanup_processes,
    initialize_agent,
    initialize_robots,
    initialize_sensors,
    setup_can_interfaces,
    setup_logging,
)
from robots_realtime.utils.portal_utils import Client


@dataclass
class LaunchConfig:
    hz: float = 30.0
    cameras: Dict[str, Tuple[CameraDriver, int]] = field(default_factory=dict)
    robots: Dict[str, Union[str, Robot]] = field(default_factory=dict)
    max_steps: Optional[int] = None  # this is for testing
    save_path: Optional[str] = None
    station_metadata: Dict[str, str] = field(default_factory=dict)
    reset_pos: Optional[List[float]] = None  # Reset position for robots (arm joints only, gripper preserved)
    home_pos: Optional[List[float]] = None  # Home position for robots (arm joints only, gripper preserved)
    dataset_observation_dir: Optional[str] = None  # Directory containing dataset observations for inference testing (optional)
    action_threshold: float = 0.05  # Threshold for action comparison when using dataset observations

@dataclass
class Args:
    config_path: Tuple[str, ...] = ("configs/yam_viser_bimanual.yaml",)


def main(args: Args) -> None:
    """
    Main launch entrypoint.

    1. Load configuration from yaml file
    2. Initialize sensors (cameras, force sensors, etc.)
    3. Setup CAN interfaces (for YAM communication)
    4. Initialize robots (hardware interface)
    5. Initialize agent (e.g. teleoperated control, policy control, etc.)
    6. Create environment
    7. Run control loop
    """
    # Setup logging and get logger
    logger = setup_logging()
    logger.info("Starting realtime control system...")

    server_processes = []
    env = None
    agent = None

    try:
        logger.info("Loading configuration...")
        configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])
        agent_cfg = configs_dict.pop("agent")
        sensors_cfg = configs_dict.pop("sensors", None)
        main_config = instantiate(configs_dict)

        logger.info("Initializing sensors...")
        camera_dict, camera_info = initialize_sensors(sensors_cfg, server_processes)

        setup_can_interfaces()

        logger.info("Initializing robots...")
        robots = initialize_robots(main_config.robots, server_processes)

        agent = initialize_agent(agent_cfg, server_processes)

        logger.info("Creating robot environment...")
        frequency = main_config.hz
        rate = Rate(frequency, rate_name="control_loop")

        # Convert positions from list to numpy array if provided
        # Handle both flat list and nested list (for multi-robot configs)
        reset_pos_array = None
        home_pos_array = None
        if main_config.reset_pos is not None:
            reset_pos_array = np.array(main_config.reset_pos)
        if main_config.home_pos is not None:
            home_pos_array = np.array(main_config.home_pos)
        
        # Use DatasetObservationEnv if dataset_observation_dir is provided, otherwise use RobotEnv
        if main_config.dataset_observation_dir is not None:
            logger.info(f"Using DatasetObservationEnv with dataset: {main_config.dataset_observation_dir}")
            env = DatasetObservationEnv(
                robot_dict=robots,
                camera_dict=camera_dict,
                control_rate_hz=rate,
                reset_pos=reset_pos_array,
                home_pos=home_pos_array,
                dataset_observation_dir=main_config.dataset_observation_dir,
                action_threshold=main_config.action_threshold,
            )
        else:
            env = RobotEnv(
                robot_dict=robots,
                camera_dict=camera_dict,
                control_rate_hz=rate,
                reset_pos=reset_pos_array,
                home_pos=home_pos_array,
            )

        logger.info("Starting control loop...")
        _run_control_loop(env, agent, main_config)
    
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt (Ctrl+C), shutting down gracefully...")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise e
    finally:
        # Cleanup - this will always run, even on KeyboardInterrupt
        logger.info("[main] Shutting down... ")
        env.close()


def _run_control_loop(env: RobotEnv, agent: Agent, config: LaunchConfig) -> None:
    """
    Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent instance
        config: Configuration object
    """
    logger = logging.getLogger(__name__)
    steps = 0
    start_time = time.time()
    loop_count = 0
    reset_pos = agent.get_initial_state()
    obs = env.reset(reset_pos=reset_pos, duration=2.0)
    logger.info(f"Action spec: {env.action_spec()}")

    # Main control loop
    while True:
        action = agent.act(obs)
        if isinstance(action, dict):
            obs = env.step(action)
            steps += 1
        elif isinstance(action, list):
            t1 = time.time()
            for i, a in enumerate(action):
                obs = env.step(a, metadata={"strict_rate": i!=0})
                steps += 1
            t2 = time.time()
            print("Execution chunk of length ", len(action), " took ", t2 - t1, " seconds")
        loop_count += 1

        elapsed_time = time.time() - start_time
        if elapsed_time >= 1:
            calculated_frequency = loop_count / elapsed_time
            logger.info(f"Control loop frequency: {calculated_frequency:.2f} Hz")
            start_time = time.time()
            loop_count = 0

        if config.max_steps is not None and steps >= config.max_steps:
            logger.info(f"Reached max steps ({config.max_steps}), stopping...")
            agent.compare_replay()
            break



if __name__ == "__main__":
    main(tyro.cli(Args))
