import pyrealsense2 as rs
import numpy as np
import cv2

ctx = rs.context()
devices = ctx.query_devices()

print(f"Found {len(devices)} devices")
for i, dev in enumerate(devices):
    print(f"Device {i}:")
    print("  Name:", dev.get_info(rs.camera_info.name))
    print("  Serial:", dev.get_info(rs.camera_info.serial_number))

# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()
# config.enable_device("230322277156")
# config.enable_device("230322271210")
config.enable_device("230322274714")
# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Show image
        cv2.imshow("RealSense Color", color_image)

        # Press ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()