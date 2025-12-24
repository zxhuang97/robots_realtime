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
        self.left_actions = np.load(os.path.join(self.episode_dir, "action-left-pos.npy"))
        self.right_actions = np.load(os.path.join(self.episode_dir, "action-right-pos.npy"))
        
        assert len(self.left_actions) == len(self.right_actions), \
            f"Action lengths mismatch: left {len(self.left_actions)}, right {len(self.right_actions)}"
        
        self.num_steps = len(self.left_actions)
        self.current_step = 0
        print(f"ReplayAgent initialized with {self.num_steps} steps from {self.episode_dir}")

    def act(self, obs: Dict[str, Any]) -> Any:
        """Returns the next action from the dataset."""
        if self.current_step >= self.num_steps:
            if self.loop:
                self.current_step = 0
                print("ReplayAgent: Looping back to start")
            else:
                # Return the last action if we're done and not looping
                # Or could raise StopIteration? But real-time loop might crash.
                # Safer to hold the last pose.
                self.current_step = self.num_steps - 1
        
        left_action = self.left_actions[self.current_step]
        right_action = self.right_actions[self.current_step]
        
        self.current_step += 1
        
        return {
            "left": {"pos": left_action},
            "right": {"pos": right_action},
        }

    @remote(serialization_needed=True)
    def action_spec(self) -> ActionSpec:
        """Define the action specification based on loaded data."""
        return {
            "left": {"pos": Array(shape=self.left_actions.shape[1:], dtype=self.left_actions.dtype)},
            "right": {"pos": Array(shape=self.right_actions.shape[1:], dtype=self.right_actions.dtype)},
        }

