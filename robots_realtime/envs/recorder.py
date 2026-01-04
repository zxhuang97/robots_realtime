"""Async observation recorder for saving observations and images during evaluation."""

import os
import pickle
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class AsyncObservationRecorder:
    """Async recorder that saves observations as pickles and concatenated camera images as PNGs."""
    
    def __init__(self, save_dir: str, camera_names: List[str]):
        self.save_dir = save_dir
        self.camera_names = camera_names
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Create save directories
        self.obs_dir = os.path.join(save_dir, "observations")
        self.img_dir = os.path.join(save_dir, "images")
        os.makedirs(self.obs_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
    
    def start(self) -> None:
        """Start the background saving thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
    
    def _worker(self) -> None:
        """Background worker that processes the save queue."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
                self._save_item(item)
                self._queue.task_done()
            except queue.Empty:
                continue
    
    def _save_item(self, item: Tuple[int, Dict[str, Any]]) -> None:
        """Save a single observation item."""
        step, obs = item
        
        # Save pickle
        pickle_path = os.path.join(self.obs_dir, f"{step:06d}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(obs, f)
        
        # Extract and concatenate camera images
        images = []
        for cam_name in self.camera_names:
            if cam_name in obs and "images" in obs[cam_name]:
                img = obs[cam_name]["images"]["rgb"]
                images.append(img)
        
        if images:
            # Concatenate images horizontally
            concat_img = np.concatenate(images, axis=1)
            img_path = os.path.join(self.img_dir, f"{step:06d}.jpg")
            # Convert RGB to BGR for cv2
            cv2.imwrite(img_path, cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))
    
    def record(self, step: int, obs: Dict[str, Any]) -> None:
        """Queue an observation for async saving."""
        self._queue.put((step, obs))
    
    def stop(self) -> None:
        """Stop the recorder and wait for pending saves to complete."""
        # Wait for queue to empty
        self._queue.join()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
    
    def __enter__(self) -> "AsyncObservationRecorder":
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        self.stop()

