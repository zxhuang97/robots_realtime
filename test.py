from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np
import time
# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
print("Connecting to policy server...")
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
print("Connected successfully!")

for step in range(1000):
    t1 = time.time()
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/top_camera-images-rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_camera-images-rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_camera-images-rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),

        "observation/state": np.random.rand(14),
        "prompt": "pick up plastic bottles and place them in the bin",
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    print(f"Step {step}: Calling inference...")
    result = client.infer(observation)
    action_chunk = result["actions"]
    print(f"Step {step}: Got action chunk of shape {action_chunk.shape}")
    t2 = time.time()
    print(f"Time taken to get action chunk: {t2 - t1}")
    frequency = 1 / (t2 - t1)
    print(f"Frequency: {frequency} Hz")