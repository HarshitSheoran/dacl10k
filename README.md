# dacl10k

All of the data can be created from EDA.ipynb codes

MMSegmentation models (Both Convnext and EVA) are trained with the config files respectively placed in their folder

To run the run.py files:

CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 run.py

The code is currently messy and might need a lot of changes in the respect of paths