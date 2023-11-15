# dacl10k

All of the data can be created from EDA.ipynb codes

MMSegmentation models (Both Convnext and EVA) are trained with the config files respectively placed in their folder

Train the configs:
mmsegmentation/work_dirs/convnext_large_exp_002/upernet_convnext_large_fp16_640x640_160k_ade20k.py
EVA/EVA-02/seg/work_dirs/upernet_eva02_large_24_512_slide_80k.py


To run the run.py files:

CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 run.py

The code is currently messy and might need a lot of changes in the respect of paths
