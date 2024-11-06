# ComfyUI_InstantIR_Wrapper
You can InstantIR to Fix blurry photos in ComfyUI ，[InstantIR](https://github.com/instantX-research/InstantIR):Blind Image Restoration with Instant Generative Reference

----

1.Installation  
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_InstantIR_Wrapper.git
```
2.requirements  
----
```
pip install -r requirements.txt
```

3.checkpoints 
3.1 
any SDXL checkpoint  
3.2  
[InstantIR](https://huggingface.co/InstantX/InstantIR)  
```
├── ComfyUI/models/InstantIR/models
|     ├── adapter.pt
|     ├── aggregator.pt
|     ├──previewer_lora_weights.bin
```
3.3 dino   
online or any local path  
[dinov2-large](https://huggingface.co/facebook/dinov2-large)  
3.4 lcm lora 
[lcm-lora-sdxl](https://huggingface.co/latent-consistency/lcm-lora-sdxl)  

----

4. Example
----      
![](https://github.com/smthemex/ComfyUI_InstantIR/blob/main/example.png)


5.Citation
------

**instantX-research/InstantIR**
``` python  
@article{huang2024instantir,
  title={InstantIR: Blind Image Restoration with Instant Generative Reference},
  author={Huang, Jen-Yuan and Wang, Haofan and Wang, Qixun and Bai, Xu and Ai, Hao and Xing, Peng and Huang, Jen-Tse},
  journal={arXiv preprint arXiv:2410.06551},
  year={2024}
}```
