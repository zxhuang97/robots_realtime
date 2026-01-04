"""
Keyboard control for robot control loop.

Commands:
    p: Pause - stop moving and enter ipdb debugger
    r: Reset - stop moving and reset arms to home position  
    q: Terminate - close env gracefully and exit
"""

import select
import sys
import termios
import tty
from enum import Enum, auto
from typing import Optional
import ipdb

class ControlCommand(Enum):
    """Available control commands."""
    NONE = auto()
    PAUSE = auto()
    RESET = auto()
    TERMINATE = auto()


class KeyboardController:
    """Non-blocking keyboard input controller for robot control loop."""
    
    KEY_MAPPINGS = {
        'p': ControlCommand.PAUSE,
        'r': ControlCommand.RESET,
        'q': ControlCommand.TERMINATE,
    }
    
    def __init__(self):
        self._old_settings: Optional[list] = None
        self._active = False
        
    def start(self):
        """Start keyboard listening - sets terminal to raw mode."""
        if sys.stdin.isatty():
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._active = True
            print("\n[Keyboard Control] Active - Press: p=pause, r=reset, q=quit\n")
    
    def stop(self):
        """Stop keyboard listening - restore terminal settings."""
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None
        self._active = False

        ipdb.set_trace()
    
    def check_input(self) -> ControlCommand:
        """
        Check for keyboard input (non-blocking).
        
        Returns:
            ControlCommand indicating what action to take
        """
        if not self._active:
            return ControlCommand.NONE
            
        # Check if there's input available (non-blocking)
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1).lower()
            if key in self.KEY_MAPPINGS:
                return self.KEY_MAPPINGS[key]
        
        return ControlCommand.NONE
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.stop()
        return False

