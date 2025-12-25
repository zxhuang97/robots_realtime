import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver


@dataclass
class RealSenseCamera(CameraDriver):
    """
    RealSense camera driver using pyrealsense2.
    
    Configured using serial_number to identify the camera.
    Serial numbers:
    - 230322271210: left
    - 230322274714: top
    - 230322277156: right
    """

    serial_number: str
    camera_type: str = "realsense_camera"
    image_transfer_time_offset: int = 80  # ms typical transfer time, can change based on computer type and load
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    name: Optional[str] = None
    intrinsic_data: Optional[dict] = None
    warmup_frames: int = 60  # Number of frames to skip for camera stabilization

    def __repr__(self) -> str:
        return f"RealSenseCamera(serial_number={self.serial_number!r}, name={self.name!r}, resolution={self.resolution}, fps={self.fps})"

    def __post_init__(self):
        # Create pipeline and config
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable device by serial number
        self.config.enable_device(self.serial_number)
        logging.info(f"RealSense camera configured with serial number: {self.serial_number}")
        
        # Configure color stream
        width, height = self.resolution
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.fps)
        
        # Start streaming
        profile = self.pipeline.start(self.config)
        logging.info(f"RealSense camera started")
        
        # Get the color stream profile for intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        # Warmup: skip initial frames to allow camera to stabilize
        logging.info(f"Skipping {self.warmup_frames} frames for camera stabilization...")
        for _ in range(self.warmup_frames):
            self.pipeline.wait_for_frames()
        
        # Store intrinsics if needed
        if self.intrinsic_data is None:
            self.intrinsic_data = {
                "rgb": {
                    "K": np.array([
                        [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                        [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                        [0, 0, 1]
                    ]),
                    "D": np.array(self.color_intrinsics.coeffs),
                    "width": width,
                    "height": height,
                }
            }
        
        logging.info(f"RealSense camera initialized: {self}")

    @classmethod
    def list_cameras(cls) -> List[Dict[str, str]]:
        """List all available RealSense cameras.
        
        Returns:
            List of dicts with 'serial_number', 'name', and 'index'
        """
        ctx = rs.context()
        devices = ctx.query_devices()
        cameras = []
        
        for i, dev in enumerate(devices):
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            cameras.append({
                "serial_number": serial,
                "name": name,
                "index": i,
            })
            logging.info(f"Found RealSense camera {i}: {name} (Serial: {serial})")
        
        return cameras

    def read(self) -> CameraData:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            logging.error(f"{self}: Failed to get color frame")
            raise RuntimeError(f"{self}: Failed to get color frame from RealSense camera")
        
        # Get frame data and convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # RealSense returns BGR, convert to RGB to match OpencvCamera API
        frame_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        
        # Get timestamp (in milliseconds)
        capture_time_ms = time.time() * 1000
        timestamp_ms = capture_time_ms - self.image_transfer_time_offset
        
        return CameraData(images={"rgb": frame_rgb}, timestamp=timestamp_ms)

    def get_camera_info(self) -> dict:
        info = {
            "camera_type": self.camera_type,
            "serial_number": self.serial_number,
            "width": self.resolution[0],
            "height": self.resolution[1],
            "fps": self.fps,
            "name": self.name if self.name is not None else "realsense_camera",
        }
        return info

    def stop(self) -> None:
        """Stop the camera."""
        self.pipeline.stop()
        logging.info(f"Stopped RealSense camera: {self}")

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        """Read calibration data from the camera.
        
        Returns:
            dict: The calibration data with intrinsics matrix and distortion coefficients.
        """
        if self.intrinsic_data is None:
            raise RuntimeError("Intrinsic data not initialized")
        return self.intrinsic_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--serial_number", type=str, required=True, help="RealSense camera serial number")
    parser.add_argument("--list_cameras", action="store_true", help="List available RealSense cameras")
    parser.add_argument("--show_video", action="store_true", help="Show video feed")
    args = parser.parse_args()

    if args.list_cameras:
        RealSenseCamera.list_cameras()
    else:
        camera = RealSenseCamera(serial_number=args.serial_number)
        
        if args.show_video:
            from robots_realtime.sensors.cameras.camera_utils import plot_camera_read
            plot_camera_read(camera)
        else:
            while True:
                data = camera.read()
                print(f"Timestamp: {data.timestamp}, Image shape: {data.images['rgb'].shape if data.images['rgb'] is not None else 'None'}")
                time.sleep(1 / camera.fps)
        
        camera.stop()

