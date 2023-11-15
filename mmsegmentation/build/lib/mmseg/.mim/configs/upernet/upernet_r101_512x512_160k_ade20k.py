_base_ = './upernet_r50_512x512_160k_ade20k.py'
#model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
model = dict(pretrained="/mnt/md0/dacl10k/AAA_MMSEG/mmsegmentation/work_dirs/upernet_r101_t1/iter_72000_backbone.pth", backbone=dict(depth=101))

fp16 = dict(loss_scale='dynamic')