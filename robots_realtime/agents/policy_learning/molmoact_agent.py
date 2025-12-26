import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from dm_env.specs import Array

from robots_realtime.agents.agent import PolicyAgent
from robots_realtime.agents.constants import ActionSpec
from robots_realtime.utils.portal_utils import remote

torch.set_float32_matmul_precision('high')


def convert_to_hf(
    checkpoint_dir: str,
    output_dir: str,
    style: str = "demo",
    chat_template: str = "demo",
    norm_stats_path: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """Convert MolmoAct checkpoint to HuggingFace format."""
    cmd = [
        "python3",
        "-m",
        "olmo.hf_model.molmoact.convert_molmoact_to_hf",
        "--checkpoint_dir",
        checkpoint_dir,
        "--output_dir",
        output_dir,
        "--style",
        style,
        "--chat_template",
        chat_template,
    ]
    
    if norm_stats_path:
        cmd.extend(["--norm_stats_path", norm_stats_path])
    
    print(f"Converting checkpoint to HuggingFace format...")
    print(f"  Input:  {checkpoint_dir}")
    print(f"  Output: {output_dir}")
    
    if dry_run:
        print(f"[DRY RUN] Command to execute:")
        print(" ".join(cmd))
        return True
    
    result = subprocess.run(cmd)
    return result.returncode == 0


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


def prepare_inputs(img, wrist_img, language_instruction, processor, device):
    """Prepare inputs for the model."""
    image = Image.fromarray(img)
    wrist = Image.fromarray(wrist_img)
    imgs = [image, wrist]

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


def generate_and_predict_optimized(model, inputs, processor, state, unnorm_key, debug=False):
    """Generate tokens and predict actions using optimized path."""
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
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_output = model.generate(
                **inputs, 
                max_new_tokens=48, 
                output_hidden_states=False, 
                return_dict_in_generate=True,
                cache_implementation="static",
                disable_compile=True
            )
            generated_ids = generation_output.sequences
    
    generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
    generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    if debug:
         print(f"generated text: {generated_text}")

    if getattr(model.config, "use_action_expert", False):
        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(state).float().to(model.device).unsqueeze(0)

        total_len = generated_ids.shape[1] - 1
        full_hidden_states = model.hidden_states_buffer[:, :total_len, :]
        
        optimized_actions = compute_action_optimized(model, full_hidden_states, inputs, state_tensor)
        
        raw_actions = optimized_actions[0]
        unnormalized_actions = model.unnormalize_action_tensor(raw_actions, unnorm_key=unnorm_key)
        
        action = unnormalized_actions.float().cpu().numpy()
        if debug:
            print(f"generated action (expert): {action}")
    else:
        action = model.parse_action(generated_text, unnorm_key=unnorm_key)
        if debug:
            print(f"generated action: {action}")
            
    return action


@dataclass
class MolmoActAgent(PolicyAgent):
    """Agent that uses MolmoAct model for action prediction."""
    
    checkpoint: str = ""
    task_description: str = "manipulate objects"
    unnorm_key: str = "xdof_data"
    style: str = "demo"
    chat_template: str = "demo"
    main_camera: str = "top_camera"
    wrist_camera: str = "left_camera"
    debug: bool = False
    action_chunk: int = 8
    norm_stats_path: Optional[str] = None
    
    def __post_init__(self):
        """Load model and processor."""
        super().__post_init__()
        
        if not self.checkpoint:
            raise ValueError("checkpoint must be provided")
        
        ckpt = self.checkpoint
        
        # Check if checkpoint needs conversion
        if not os.path.exists(os.path.join(ckpt, "config.json")):
            hf_ckpt = f"{ckpt.rstrip('/')}-hf"
            if not os.path.exists(os.path.join(hf_ckpt, "config.json")):
                print(f"Config not found in {ckpt} or {hf_ckpt}. Converting...")
                success = convert_to_hf(
                    checkpoint_dir=ckpt,
                    output_dir=hf_ckpt,
                    style=self.style,
                    chat_template=self.chat_template,
                    norm_stats_path=self.norm_stats_path,
                    dry_run=False
                )
                if not success:
                    raise ValueError(f"Failed to convert checkpoint {ckpt}")
            
            print(f"Using HF checkpoint: {hf_ckpt}")
            ckpt = hf_ckpt
        
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
            attn_implementation="sdpa",
        )
        
        # Determine unnorm_key if not in norm_stats
        if self.unnorm_key not in self.model.norm_stats:
            print(f"unnorm_key {self.unnorm_key} not found in model.norm_stats")
            self.unnorm_key = list(self.model.norm_stats.keys())[0]
            print(f"using default unnorm_key: {self.unnorm_key}")
    
    def _extract_images(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract main and wrist images from observations."""
        # Try to get main camera image
        main_img = None
        if self.main_camera in obs and "images" in obs[self.main_camera]:
            main_img = obs[self.main_camera]["images"].get("rgb")
        
        # Try to get wrist camera image
        wrist_img = None
        if self.wrist_camera in obs and "images" in obs[self.wrist_camera]:
            wrist_img = obs[self.wrist_camera]["images"].get("rgb")
        
        # Fallback: use main camera for both if wrist not available
        if main_img is None:
            # Try other common camera names
            for cam_name in ["top_camera", "left_camera", "right_camera"]:
                if cam_name in obs and "images" in obs[cam_name]:
                    main_img = obs[cam_name]["images"].get("rgb")
                    if main_img is not None:
                        break
        
        if wrist_img is None:
            # Use main image as wrist image if not available
            wrist_img = main_img
        
        if main_img is None or wrist_img is None:
            raise ValueError(f"Could not extract images from observation. Available keys: {list(obs.keys())}")
        
        # Ensure images are uint8 numpy arrays
        if not isinstance(main_img, np.ndarray):
            main_img = np.array(main_img)
        if not isinstance(wrist_img, np.ndarray):
            wrist_img = np.array(wrist_img)
        
        if main_img.dtype != np.uint8:
            main_img = (main_img * 255).astype(np.uint8) if main_img.max() <= 1.0 else main_img.astype(np.uint8)
        if wrist_img.dtype != np.uint8:
            wrist_img = (wrist_img * 255).astype(np.uint8) if wrist_img.max() <= 1.0 else wrist_img.astype(np.uint8)
        
        return main_img, wrist_img
    
    def _extract_state(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract proprioceptive state from observations."""
        # Try to extract joint positions and gripper positions
        state_parts = []
        
        # Try left arm
        if "left" in obs:
            if "joint_pos" in obs["left"]:
                state_parts.append(obs["left"]["joint_pos"])
            if "gripper_pos" in obs["left"]:
                state_parts.append(obs["left"]["gripper_pos"])
        
        # Try right arm
        if "right" in obs:
            if "joint_pos" in obs["right"]:
                state_parts.append(obs["right"]["joint_pos"])
            if "gripper_pos" in obs["right"]:
                state_parts.append(obs["right"]["gripper_pos"])
        
        if len(state_parts) == 0:
            return None
        
        state = np.concatenate(state_parts, axis=-1)
        return state.astype(np.float32)
    
    @remote()
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate action from observation."""
        # Extract images and state
        main_img, wrist_img = self._extract_images(obs)
        state = self._extract_state(obs)
        
        # Prepare inputs
        inputs = prepare_inputs(
            main_img, 
            wrist_img, 
            self.task_description, 
            self.processor, 
            self.model.device
        )
        
        # Generate action
        action_matrix = generate_and_predict_optimized(
            self.model, 
            inputs, 
            self.processor, 
            state, 
            self.unnorm_key, 
            debug=self.debug
        )
        
        # Handle action format
        if isinstance(action_matrix, np.ndarray):
            if action_matrix.ndim == 2:
                # Action matrix: take first action_chunk actions
                action_matrix = action_matrix[:self.action_chunk]
            elif action_matrix.ndim == 1:
                # Single action vector
                action_matrix = action_matrix[np.newaxis, :]
        
        # Extract first action
        if isinstance(action_matrix, (list, tuple)):
            first_action = action_matrix[0]
        else:
            first_action = action_matrix[0] if action_matrix.ndim > 1 else action_matrix
        
        # Convert to expected format
        # Assuming action is 7D: [x, y, z, qx, qy, qz, qw, gripper] or similar
        # For bimanual, might be 14D: [left_7d, right_7d]
        action_dim = len(first_action) if hasattr(first_action, '__len__') else 1
        
        if action_dim == 7:
            # Single arm
            return {
                "left": {"pos": first_action[:7]}
            }
        elif action_dim == 14:
            # Bimanual
            return {
                "left": {"pos": first_action[:7]},
                "right": {"pos": first_action[7:14]}
            }
        else:
            # Fallback: return as-is and let environment handle it
            return {"action": first_action}
    
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

