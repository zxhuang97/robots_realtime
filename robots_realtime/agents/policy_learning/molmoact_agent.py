import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from dm_env.specs import Array

from robots_realtime.agents.agent import PolicyAgent
from robots_realtime.agents.constants import ActionSpec
from robots_realtime.utils.portal_utils import remote
import torch.nn.functional as F
import cv2
import sys
sys.path.append("FAR_molmoact")
from olmo.transforms import Normalizer, make_bool_mask, AbsoluteActions, DeltaActions
from einops import rearrange

torch.set_float32_matmul_precision('high')

def compute_action_optimized(model, full_hidden_states, inputs, state_tensor):
    """Reuse hidden states from buffer to compute actions."""
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            # Construct vl_mask
            prompt_mask = inputs.get('attention_mask')
            if prompt_mask is None:
                prompt_len = inputs['input_ids'].shape[1]
                prompt_mask = torch.ones((full_hidden_states.shape[0], prompt_len), device=full_hidden_states.device)
            else:
                prompt_mask = prompt_mask.to(full_hidden_states.device)
                
            # The generated part is valid
            generated_mask = torch.ones((full_hidden_states.shape[0], full_hidden_states.shape[1] - prompt_mask.shape[1]), device=full_hidden_states.device)
            vl_mask = torch.cat([prompt_mask, generated_mask], dim=1).bool()
            
            # Knowledge insulation logic
            if model.config.action_expert_config.knowledge_insulation:
                x_for_action = full_hidden_states.detach()
            else:
                x_for_action = full_hidden_states
                
            predicted_actions = model.action_expert.predict_action(x_for_action, state_tensor, vl_mask)
            
    return predicted_actions

def parse_flow_tokens_from_text(text: str, expected_length: int = 15, num_views: int = 2, enforce_format=False) -> Optional[list[int]]:
    """Parse flow tokens from model output text.

    Handles format:
        - Tagged: "<FLOW_START><FLOW_12><FLOW_345>...<FLOW_END>"
        - Interleaved: Multiple "<FLOW_START>...<FLOW_END>" blocks (concatenates tokens from all blocks).

    Args:
        text: Text to parse.
        expected_length: Expected total length of flow tokens.
                         If provided, used for padding/truncation.
        num_views: Number of views expected. Used to infer block length for interleaved parsing.

    Returns:
        List of token IDs, or default tokens if parsing fails and expected_length provided.
    """
    # Extract the flow portion if it's in a longer text
    # Use findall to support interleaved mode (multiple START/END blocks)
    flow_matches = re.findall(r'<FLOW_START>.*?<FLOW_END>', text)

    if not flow_matches:
        num_valid_blocks = 0
        flow_matches = []
    else:
        num_valid_blocks = len(flow_matches)
    if enforce_format:
        for _ in range(num_views - num_valid_blocks):
            flow_matches.append("<FLOW_START>" + "<FLOW_0>" * expected_length + "<FLOW_END>")
    all_tokens = []

    def parse_block(s):
        s = s.replace("<FLOW_START>", "").replace("<FLOW_END>", "")
        s = s.replace("<FLOW_", "")
        pieces = [p for p in s.split(">") if p.strip() != ""]
        tokens = []
        try:
            tokens = [int(p) for p in pieces]
        except ValueError:
            pass
        return tokens

    for s in flow_matches:
        block_tokens = parse_block(s)
        if enforce_format:
            if len(block_tokens) < expected_length:
                block_tokens.extend([0] * (expected_length - len(block_tokens)))
            elif len(block_tokens) > expected_length:
                block_tokens = block_tokens[:expected_length]
        all_tokens.extend(block_tokens)
    return all_tokens

def prepare_inputs(imgs, language_instruction, processor, device):
    """Prepare inputs for the model."""
    imgs = [Image.fromarray(img) for img in imgs]
    prompt = (
        f"The task is {language_instruction}. What is the future motion of the image?" 
    )
        
    text = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [dict(type="text", text=prompt)]
            }
        ], 
        tokenize=False, 
        add_generation_prompt=True,
    )
    text = "User: " + text + " Assistant: The future motion of the image is "
    inputs = processor(
        images=[imgs],
        text=text,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


def generate_and_predict_optimized(model, inputs, processor, state, debug=False):
    """Generate tokens and predict actions using optimized path.
    
    Returns:
        Tuple of (n_actions, generated_text) if debug=True, else just n_actions
    """
    use_action_expert = getattr(model.config, "use_action_expert", False)
    
    if use_action_expert:
        batch_size = inputs['input_ids'].shape[0]
        buffer_len = 2048
        new_buffer = torch.zeros(
            (batch_size, buffer_len, model.config.hidden_size), 
            dtype=torch.bfloat16, 
            device=model.device
        )
        if "hidden_states_buffer" in model._buffers:
            if model.hidden_states_buffer.shape != new_buffer.shape:
                model.hidden_states_buffer.resize_as_(new_buffer)
            model.hidden_states_buffer.zero_()
        else:
            model.register_buffer("hidden_states_buffer", new_buffer, persistent=False)
    
    generated_ids = None
    t1 = time.time()
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_output = model.generate(
                **inputs, 
                max_new_tokens=75, 
                output_hidden_states=False, 
                return_dict_in_generate=True,
                # cache_implementation="static",
                disable_compile=True
            )
            generated_ids = generation_output.sequences
    t2 = time.time()
    print(f"[generation] time: {t2 - t1} seconds")
    generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
    generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    if debug:
         print(f"generated text: {generated_text}")

    state_tensor = torch.from_numpy(state).float().to(model.device).unsqueeze(0)

    total_len = generated_ids.shape[1] - 1
    full_hidden_states = model.hidden_states_buffer[:, :total_len, :]
    
    n_actions = compute_action_optimized(model, full_hidden_states, inputs, state_tensor)
    return n_actions, generated_text


@dataclass
class MolmoActAgent(PolicyAgent):
    """Agent that uses MolmoAct model for action prediction."""
    
    checkpoint: str = ""
    task_description: str = "manipulate objects"
    unnorm_key: str = "flow_tokenized"
    style: str = "demo"
    chat_template: str = "demo"
    camera_names: list[str] = field(default_factory=lambda: ["top_camera", "left_camera", "right_camera"])
    debug: bool = False
    action_chunk: int = 8
    norm_stats_path: Optional[str] = None
    motion_tokenizer_checkpoint: Optional[str] = None  # Path to motion tokenizer for flow visualization
    train_shape: Tuple[int, int] = (224, 168)
    resize_to_train_shape: bool = True
    
    def __post_init__(self):
        """Load model and processor."""
        ckpt = self.checkpoint
        
        self.processor = AutoProcessor.from_pretrained(
            ckpt,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            device_map="auto",
            padding_side="left",
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            ckpt,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            device_map="auto",
            # attn_implementation="sdpa",
        )
        
        self.normalizer = Normalizer(self.model.norm_stats[self.unnorm_key])
        
        delta_action_mask = make_bool_mask(6, -1, 6, -1)
        self.absolute_action_transform = AbsoluteActions(mask=delta_action_mask)
        self.delta_action_transform = DeltaActions(mask=delta_action_mask)

        self.l1_action_loss = []
        # Load motion tokenizer for debug visualization if provided
        self.motion_tokenizer = None
        if self.debug and self.motion_tokenizer_checkpoint:
            from amplify.models.motion_tokenizer import load_motion_tokenizer
            self.motion_tokenizer, _ = load_motion_tokenizer(self.motion_tokenizer_checkpoint, frozen=True)
            self.motion_tokenizer.eval().to(self.model.device)
            print(f"Loaded motion tokenizer from {self.motion_tokenizer_checkpoint} for debug visualization")
    
    def _extract_images(self, obs: Dict[str, Any]) -> list[np.ndarray]:
        """Extract main and wrist images from observations."""
        images = []
        for camera_name in self.camera_names:
            img = obs[camera_name]["images"]["rgb"].astype(np.uint8)
            if self.resize_to_train_shape:
                img = cv2.resize(img, self.train_shape, interpolation=cv2.INTER_LINEAR)
            images.append(img)
        return images
    
    def _extract_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract proprioceptive state from observations."""
        state_parts = []
        state_parts.append(obs["left"]["joint_pos"])
        state_parts.append(obs["left"]["gripper_pos"])
        state_parts.append(obs["right"]["joint_pos"])
        state_parts.append(obs["right"]["gripper_pos"])
        state = np.concatenate(state_parts, axis=-1)
        return state.astype(np.float32)
    
    def _decode_flow_tokens_to_tracks(self, tokens: list[int], n_tracks: int = 400, views: int = 3) -> Optional[np.ndarray]:
        """Decode flow tokens to absolute point tracks.
        
        Args:
            tokens: List of token IDs
            n_tracks: Number of tracks (should be 400)
            views: Number of views (should match number of cameras)
            
        Returns:
            Point tracks as numpy array (views, frames, n_tracks, 2) in [-1, 1] or None if decoding fails
        """
        if self.motion_tokenizer is None:
            return None
        
        if len(tokens) % self.motion_tokenizer.num_timesteps != 0:
            return None
        
        from amplify.utils.kp_utils.query_utils import grid_queries
        from amplify.utils.data_utils import velocities_to_points
        
        # Convert tokens to tensor and decode to velocities
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=self.model.device)
        with torch.no_grad():
            z_quantized = self.motion_tokenizer.quantize.indices_to_codes(token_tensor).unsqueeze(0)
            reconstructed_flow, _ = self.motion_tokenizer.decode(z_quantized)
            # Generate initial grid points: (views, n_tracks, 2) in [-1, 1]
            initial_points = grid_queries(views=views, n_tracks=n_tracks, device=self.model.device).tensor
            initial_points = initial_points.unsqueeze(0).unsqueeze(2)
            # Convert velocities to absolute tracks
            tracks = velocities_to_points(
                reconstructed_flow,
                time_dim=2,
                init_points=initial_points
            )
            tracks = tracks.squeeze(0)
        
        return tracks.cpu().numpy()
    
    def _visualize_flow_tracks(self, images: list[np.ndarray], tracks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Visualize flow tracks on images.
        
        Args:
            images: List of input images (H, W, 3) in [0, 255]
            tracks: Flow tracks (views, frames, n_tracks, 2) in [-1, 1] or None
            
        Returns:
            Visualization image as numpy array (H, W*views, 3) in [0, 255] or None
        """
        if tracks is None:
            return None
        
        from amplify.utils.vis_utils import vis_pred
        
        # Stack images: (views, H, W, 3)
        image_stack = np.stack(images, axis=0)
        image_tensor = torch.tensor(image_stack).float().unsqueeze(0) / 255.0
        
        # Visualize tracks
        tracks_tensor = torch.from_numpy(tracks).unsqueeze(0)
        vis = vis_pred(image_tensor, tracks_tensor)
        vis = vis.squeeze(0).numpy()
        
        if vis.dtype != np.uint8:
            vis = np.clip(vis, 0, 255).astype(np.uint8)
        
        return vis
    
    @remote()
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate action from observation."""
        # Extract images and state
        images = self._extract_images(obs)
        state = self._extract_state(obs)
        n_state = self.normalizer.normalize(state, key="state")
        # Prepare inputs
        inputs = prepare_inputs(
            images,
            self.task_description, 
            self.processor, 
            self.model.device
        )
        
        # Generate action
        result = generate_and_predict_optimized(
            self.model, 
            inputs, 
            self.processor, 
            n_state, 
            debug=self.debug
        )
        n_actions, generated_text = result
        n_actions = n_actions.float().cpu().numpy()[0]
        # Parse and visualize flow tokens in debug mode
        # if self.debug and generated_text:
        #     num_views = len(images)
        #     expected_len = 15  # Default expected length for flow tokens
        #     flow_tokens = parse_flow_tokens_from_text(
        #         generated_text, 
        #         expected_length=expected_len, 
        #         num_views=num_views, 
        #         enforce_format=False
        #     )
            
        #     if flow_tokens:
        #         print(f"Parsed {len(flow_tokens)} flow tokens")
        #         # Decode flow tokens to tracks
        #         tracks = self._decode_flow_tokens_to_tracks(flow_tokens, n_tracks=400, views=num_views)
        #         if tracks is not None:
        #             # Visualize tracks on images
        #             vis_image = self._visualize_flow_tracks(images, tracks)
        #             if vis_image is not None:
        #                 # Save visualization (you can modify this to display or save elsewhere)
        #                 import os
        #                 os.makedirs("debug_visualizations", exist_ok=True)
        #                 import cv2
        #                 vis_path = f"debug_visualizations/flow_vis_{int(time.time() * 1000)}.png"
        #                 cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        #                 print(f"Saved flow visualization to {vis_path}")
        #         else:
        #             print("Failed to decode flow tokens to tracks (motion tokenizer may not be loaded)")
        #     else:
        #         print("No flow tokens found in generated text")
        
        delta_action = self.normalizer.unnormalize(n_actions, key="action")
        gt_delta_action = self.delta_action_transform({"actions": obs["gt_action_chunks"], "states": state})["actions"]
        n_gt_delta_action = self.normalizer.normalize(gt_delta_action, key="action")
        diff = np.abs(n_actions - n_gt_delta_action).mean()
        self.l1_action_loss.append(diff)
        print(f"L1 action difference (normalied delta): {diff}")
        action = self.absolute_action_transform({"actions": delta_action, "state": state})["actions"]
        left_action = action[:, :7]
        right_action = action[:, 7:]
        actions_list = []
        for i in range(len(left_action)):
            actions_list.append({
                "left": {"pos": left_action[i]},
                "right": {"pos": right_action[i]},
            })
        return actions_list

    def compare_replay(self) -> None:
        """Compare replay with ground truth actions."""
        l1_action_loss = np.array(self.l1_action_loss)
        print(f"L1 action loss mean: {l1_action_loss.mean()}")
        print(f"L1 action loss max: {l1_action_loss.max()}")

    @remote(serialization_needed=True)
    def action_spec(self) -> ActionSpec:
        """Define the action specification."""
        # Default to single arm 7D action
        return {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
        }
    
    def reset(self) -> None:
        """Reset agent state."""
        pass

