# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
import logging
import folder_paths

from .utils import load_list_images, tensor2imglist, tensor_upscale,instantIR_main, instantIR_load_model, auto_downlaod

node_cur_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32
    
MAX_SEED = np.iinfo(np.int32).max

# add checkpoints dir
InstantIR_current_path = os.path.join(folder_paths.models_dir, "InstantIR")
if not os.path.exists(InstantIR_current_path):
    os.makedirs(InstantIR_current_path)

try:
    folder_paths.add_model_folder_path("InstantIR", InstantIR_current_path, False)
except:
    folder_paths.add_model_folder_path("InstantIR", InstantIR_current_path)

if os.path.exists(folder_paths.cache_dir):
    InstantIR_current_path = os.path.join(folder_paths.cache_dir, "InstantIR")
    try:
        folder_paths.add_model_folder_path("InstantIR", InstantIR_current_path, False)
    except:
        folder_paths.add_model_folder_path("InstantIR", InstantIR_current_path)

InstantIR_base_path = os.path.join(InstantIR_current_path, "models")  # InstantIR/models
if not os.path.exists(InstantIR_base_path):
    os.makedirs(InstantIR_base_path)

InstantIR_dino_path = os.path.join(InstantIR_current_path, "dino")  # InstantIR/models
if not os.path.exists(InstantIR_dino_path):
    os.makedirs(InstantIR_dino_path)


class InstantIR_Loader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        INSTANT_LIST=["none"]+[i for i in folder_paths.get_filename_list("InstantIR") if ".bin" in i  or ".pt" in i] if folder_paths.get_filename_list("InstantIR") else ["none"]
        return {
            "required": {
                "sdxl_checkpoints": (["none"] + folder_paths.get_filename_list("checkpoints"),),
                "dino_repo": ("STRING", {"default": "facebook/dinov2-large"}),
                "adapter_checkpoints": (INSTANT_LIST,),
                "aggregator_checkpoints": (INSTANT_LIST,),
                "lora": (["none"] + folder_paths.get_filename_list("loras"),),
                "InstantIR_lora": (INSTANT_LIST,),
                "use_clip_encoder": ("BOOLEAN", {"default": True},),
                "low_vram": ("BOOLEAN", {"default": False},),
            }
        }
    
    RETURN_TYPES = ("InstantIR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main_"
    CATEGORY = "InstantIR"
    
    def main_(self, sdxl_checkpoints, dino_repo, adapter_checkpoints, aggregator_checkpoints,lora, InstantIR_lora,
             use_clip_encoder,low_vram):
        if not dino_repo:
            logging.info("no dino files in dir ,auto download from facebook/dinov2-large")
            vision_encoder_path = auto_downlaod(InstantIR_current_path, "dino")
        else:
            vision_encoder_path = dino_repo
        
        if os.path.exists(os.path.join(InstantIR_current_path, "dino")):
            vision_encoder_path = os.path.join(InstantIR_current_path, "dino")
        
        if sdxl_checkpoints != "none":
            sdxl_path = folder_paths.get_full_path("checkpoints", sdxl_checkpoints)
        else:
            #sdxl_path='stabilityai/stable-diffusion-xl-base-1.0'
            raise "need chocie a sdxl checkpoint"
        
        if adapter_checkpoints != "none":
            adapter_path = folder_paths.get_full_path("InstantIR", adapter_checkpoints)
        else:
            raise "need chocie a adapter checkpoint"
        
        if InstantIR_lora != "none":
            previewer_lora_path = folder_paths.get_full_path("InstantIR", InstantIR_lora)
        else:
            raise "need chocie a lora checkpoint"
        
        if lora != "none":
            lora_path = folder_paths.get_full_path("loras", lora)
        else:
            raise "need chocie a lora checkpoint"
        
        if aggregator_checkpoints != "none":
            aggregator_path = folder_paths.get_full_path("InstantIR", aggregator_checkpoints)
        else:
            raise "need chocie a aggregator checkpoint"
        
        model = instantIR_load_model(use_clip_encoder, vision_encoder_path, sdxl_path, adapter_path,
                                     previewer_lora_path,lora_path, aggregator_path, device)
        
        logging.info("loading checkpoint done.")
        
        if low_vram and not torch.backends.mps.is_available():
            model.to(dtype=torch_dtype)
            model.enable_model_cpu_offload() #MPS not support
        else:
            model.to(device=device, dtype=torch_dtype)
        return (model,)


class InstantIR_Sampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("InstantIR_MODEL",),
                "pixels": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Photorealistic, highly detailed, hyper detailed photo - realistic maximum detail, 32k, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations, taken using a Canon EOS R camera, Cinematic, High Contrast, Color Grading. "}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": "blurry, out of focus, unclear, depth of field, over-smooth, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, dirty, messy, worst quality, low quality, frames, painting, illustration, drawing, art, watermark, signature, jpeg artifacts, deformed, lowres"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.01}),
                "creative_restoration": ("BOOLEAN", {"default": False},),
                "width": ("INT", {"default": 768, "min": 128, "max": 1024, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 768, "min": 128, "max": 1024, "step": 64, "display": "number"}),
                "preview_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,}),
                "guidance_end": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 30.0, "step": 0.1,}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1, }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "InstantIR"
    
    def main(self, model, pixels, prompt, negative_prompt, seed, steps, cfg, creative_restoration, width, height, preview_start,guidance_end,batch_size):
        # pre image
        image = tensor_upscale(pixels, width, height)
        image_list, _ = tensor2imglist(image, np_out=True)
        
        logging.info(f"Start infer {len(image_list)} images.")
        ouput_img = instantIR_main(image_list, model, seed, creative_restoration, steps, prompt, negative_prompt, cfg,
                                   batch_size, device,preview_start,guidance_end)
        logging.info("finish processing")
        image = load_list_images(ouput_img)
        
        return (image,)


NODE_CLASS_MAPPINGS = {
    "InstantIR_Loader": InstantIR_Loader,
    "InstantIR_Sampler": InstantIR_Sampler,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantIR_Loader": "InstantIR_Loader",
    "InstantIR_Sampler": "InstantIR_Sampler",
}
