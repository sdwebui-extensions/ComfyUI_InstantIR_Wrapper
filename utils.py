# !/usr/bin/env python
# -*- coding: UTF-8 -*-
from PIL import Image
import cv2
import os
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from .schedulers.lcm_single_step_scheduler import LCMSingleStepScheduler
from diffusers import DDPMScheduler

from .pipelines.sdxl_instantir import InstantIRPipeline
from .module.ip_adapter.utils import load_adapter_to_pipe
from comfy.utils import common_upscale,ProgressBar

if torch.cuda.is_available():
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32
    
cur_path = os.path.dirname(os.path.abspath(__file__))

def name_unet_submodules(unet):
    def recursive_find_module(name, module, end=False):
        if end:
            for sub_name, sub_module in module.named_children():
                sub_module.full_name = f"{name}.{sub_name}"
            return
        if not "up_blocks" in name and not "down_blocks" in name and not "mid_block" in name: return
        elif "resnets" in name: return
        for sub_name, sub_module in module.named_children():
            end = True if sub_name == "transformer_blocks" else False
            recursive_find_module(f"{name}.{sub_name}", sub_module, end)

    for name, module in unet.named_children():
        recursive_find_module(name, module)

def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        # ratio = min_side / min(h, w)
        # w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def calc_mean_std(feat, eps=1e-5):
	"""Calculate mean and std for adaptive_instance_normalization.
	Args:
		feat (Tensor): 4D tensor.
		eps (float): A small value added to the variance to avoid
			divide-by-zero. Default: 1e-5.
	"""
	size = feat.size()
	assert len(size) == 4, 'The input feature should be 4D tensor.'
	b, c = size[:2]
	feat_var = feat.view(b, c, -1).var(dim=2) + eps
	feat_std = feat_var.sqrt().view(b, c, 1, 1)
	feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
	return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def auto_downlaod(current_path,repo):
    if repo=="dino":
        dino_path=os.path.join(current_path,repo)
        for i in ["model.safetensors","config.json","preprocessor_config.json"]:
            if not os.path.exists(os.path.join(dino_path,i)):
                print(f"{i} in {dino_path} ,try download from huggingface!")
                hf_hub_download(
                    repo_id="facebook/dinov2-large",
                    subfolder="",
                    filename=i,
                    local_dir=f"{current_path}/{repo}",
                )
        return dino_path
        
def instantIR_load_model(use_clip_encoder,vision_encoder_path,sdxl_path,adapter_path,previewer_lora_path,lora_path,aggregator_path,device):
    
    # Base models.
 
    modle_config = os.path.join(cur_path, "config_files/sdxl_repo")
    original_config_file = os.path.join(cur_path, "config_files/sd_xl_base.yaml")
    try:
        pipe = InstantIRPipeline.from_single_file(
            sdxl_path, config=modle_config, original_config=original_config_file, torch_dtype=torch.float16,use_clip_encoder=use_clip_encoder,)
    except:
        pipe = InstantIRPipeline.from_single_file(
                sdxl_path, config=modle_config, original_config_file=original_config_file,
                torch_dtype=torch.float16,use_clip_encoder=use_clip_encoder,)
    
    
    # Image prompt projector.
    load_adapter_to_pipe(
        pipe,
        adapter_path,
        vision_encoder_path,
    )
    
    # Prepare previewer
    lora_alpha = pipe.prepare_previewers(previewer_lora_path)
    print(f"use lora alpha {lora_alpha}")
    
    #lora_alpha = pipe.prepare_previewers("latent-consistency/lcm-lora-sdxl", use_lcm=True)
    lora_alpha = pipe.prepare_previewers(lora_path, use_lcm=True)
    print(f"use lcm lora alpha {lora_alpha}")
    
    
    pipe.scheduler = DDPMScheduler.from_pretrained(modle_config, subfolder="scheduler")
    #lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)
    
    # Load weights.
    print("Loading checkpoint...")
    pretrained_state_dict = torch.load(aggregator_path, map_location="cpu")
    pipe.aggregator.load_state_dict(pretrained_state_dict,strict=False)
    pipe.aggregator.to(device, dtype=torch_dtype)
    
    del pretrained_state_dict
    torch.cuda.empty_cache()
    return pipe

def instantIR_main(lq,pipe, seed,creative_restoration,num_inference_steps,prompt,neg_prompt,cfg,batch_size,device,preview_start,guidance_end):
    
    lcm_scheduler = LCMSingleStepScheduler.from_config(pipe.scheduler.config)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    if creative_restoration:
        if "lcm" not in pipe.unet.active_adapters():
            pipe.unet.set_adapter('lcm')
    else:
        if "previewer" not in pipe.unet.active_adapters():
            pipe.unet.set_adapter('previewer')
    
    if guidance_end > 1.0:
        guidance_end = guidance_end / num_inference_steps

    if preview_start > 1.0:
        preview_start = preview_start / num_inference_steps
    
    timesteps = [
        i * (1000 // num_inference_steps) + pipe.scheduler.config.steps_offset for i in range(0, num_inference_steps)
    ]
    timesteps = timesteps[::-1]

    if not isinstance(prompt, list):
        prompt = [prompt]
    prompt = prompt*len(lq)

    if not isinstance(neg_prompt, list):
        neg_prompt = [neg_prompt]
    neg_prompt = neg_prompt*len(lq)
    img_list=[]
    for i in range(batch_size):
        image = pipe(
            prompt=prompt,
            image=lq,
            num_inference_steps=num_inference_steps,
            generator=generator,
            timesteps=timesteps,
            negative_prompt=neg_prompt,
            guidance_scale=cfg,
            previewer_scheduler=lcm_scheduler,
            preview_start=preview_start,
            control_guidance_end=guidance_end,
        )[0]
        
        img_list.append(image) #image:list
    iamge_list=[]
    for j in img_list: # [list,list]
        for i in j:
           iamge_list.append(i)
         
    return iamge_list

def spilit_tensor2list(img_tensor):#[B,H,W,C], C=3,B>=1
    video_list = []
    if isinstance(img_tensor, list):
        if isinstance(img_tensor[0], torch.Tensor):
            video_list = img_tensor
    elif isinstance(img_tensor, torch.Tensor):
        b, _, _, _ = img_tensor.size()
        if b == 1:
            img = [b]
            while img is not []:
                video_list += img
        else:
            video_list = torch.chunk(img_tensor, chunks=b)
    return video_list
    
def tensor2imglist(image,np_out=True):# pil first
    B, _, _, _ = image.size()
    if B == 1:
        if np_out:
            list_out = [tensor2pil(image)]
        else:
            list_out = [tensor2cv(image.squeeze())]
    else:
        image_list = torch.chunk(image, chunks=B)
        if  np_out:
            list_out = [tensor2pil(i) for i in image_list]
        else:
            list_out = [tensor2cv(i.squeeze()) for i in image_list]
    return list_out,B


def tensor2pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def pil2narry(img):
    narry = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return narry

def tensor_upscale2pil(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor2pil(samples)
    return img_pil

def tensor_upscale(img_tensor, width, height): #torch tensor
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def tensor2cv(tensor_image):
    if tensor_image.is_cuda:
        tensor_image = tensor_image.detach().cpu()
    tensor_image=tensor_image.numpy()
    #反归一化
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def cvargb2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def cv2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def images_generator(img_list: list,):
    #get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_,Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_,np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    
    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in=img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in,np.ndarray):
            i=cv2.cvtColor(img_in,cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            print(i.shape)
            return i
        else:
           raise "unsupport image list,must be pil,cv2 or tensor!!!"
        
    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image

def load_list_images(img_list: list,):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images

