#!/usr/bin/env python
# coding: utf-8

# In[3]:


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default
    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)
    
    return world_size

def sync_across_gpus(tensor, local_rank, world_size):

    # print(local_rank, tensor)
    dist.barrier()
    if tensor.dim() == 0:
        gather_t_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_t_tensor, tensor)
        gather_t_tensor = torch.stack(gather_t_tensor)
    else:
        local_batch_size = torch.tensor(tensor.size(0)).cuda().int()
        all_batch_sizes = [torch.zeros(1).cuda().int()] * world_size
        dist.all_gather(all_batch_sizes, local_batch_size)
        max_batch_size = torch.stack(all_batch_sizes).max().item()

        if local_batch_size < max_batch_size:
            padding = torch.zeros((max_batch_size - local_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=0)

        dist.barrier()

        gather_t_tensor = [torch.zeros((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device) for _ in range(world_size)]
        dist.all_gather(gather_t_tensor, tensor)
        cleaned_tensors = [t[:all_batch_sizes[i]] for i, t in enumerate(gather_t_tensor)]
        gather_t_tensor = torch.cat(cleaned_tensors)

    return gather_t_tensor
    
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from glob import glob
import copy
import time
import math
import command
import random

#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = 12, 8

from skimage import img_as_ubyte
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import *
from sklearn.metrics import *

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import timm

from transformers import get_cosine_schedule_with_warmup

import torch.distributed as dist


# In[ ]:





# In[4]:


class CFG:
    DDP = 1
    DDP_INIT_DONE = 0
    N_GPUS = 3
    FOLD = 0
    FULLDATA = 0
    
    model_name = 'eva_02_large_exp_001'
    V = '5'
    
    OUTPUT_FOLDER = f"/mnt/md0/dacl10k/AAA_MMSEG/TRY1_SEG/{model_name}_v{V}"
    
    seed = 3407
    
    device = torch.device('cuda')
    
    n_folds = 4
    folds = [i for i in range(n_folds)]
    
    image_size = [512, 512]
    
    train_batch_size = 4
    valid_batch_size = 7
    acc_steps = 1
    
    lr = 1e-5
    wd = 1e-3
    n_epochs = 10
    n_warmup_steps = 0
    upscale_steps = 1.5
    validate_every = 1
    
    epoch = 0
    global_step = 0
    literal_step = 0

    autocast = True

    workers = 4

if CFG.FULLDATA:
    CFG.seed = CFG.FOLD
    
OUTPUT_FOLDER = CFG.OUTPUT_FOLDER
        
CFG.cache_dir = CFG.OUTPUT_FOLDER + '/cache/'
os.makedirs(CFG.cache_dir, exist_ok=1)
    
seed_everything(CFG.seed)


# In[ ]:





# In[5]:


training_files = glob('/mnt/md0/dacl10k/images/train/*.npy')
validation_files = glob('/mnt/md0/dacl10k/images/validation/*.npy')
files = np.concatenate([training_files, validation_files])
files[:5], files.shape


# In[ ]:





# In[6]:


class AbdDataset(Dataset):
    def __init__(self, data, transforms, is_training):
        self.data = data
        self.transforms = transforms
        self.is_training = is_training
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        
        file = self.data[i]
        
        image = np.load(file)
        mask = np.load(file.replace('/images/', '/annotations/'))
        
        #'''
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
                
            image = transformed['image']
            #image = image.float() / 255
            
            mask = transformed['mask']
            mask = mask.permute(2, 0, 1).float()
            
        #'''
        
        label = mask.sum(-1).sum(-1).clip(0, 1).float()
        
        return {'images': image,
                'masks': mask,
                'labels': label,
                'ids': f"{file}"}


# In[ ]:





# In[54]:


folds = [*KFold(n_splits=CFG.n_folds).split(files)]

def get_loaders():
    '''
    if not CFG.FULLDATA:
        valid_files = files[folds[CFG.FOLD][1]]
    else:
        valid_files = files[folds[0][1]]
    
    if not CFG.FULLDATA:
    
        train_files = files[folds[CFG.FOLD][0]]
    
        #train_volumes = [vol for vol in train_volumes if vol.liver_size.max() > 0.3]
        #valid_volumes = [vol for vol in valid_volumes if vol.liver_size.max() > 0.3]
    else:
        train_files = files
    '''
    
    train_files = training_files.copy() + validation_files[:int(len(validation_files)*0.9)].copy() 
    valid_files = validation_files[int(len(validation_files)*0.9):].copy()
    
    #train_files = np.concatenate([train_files]*5)
    
    #train_df = pd.read_csv('/mnt/md0/birdclef23/specs/train.csv')
    #valid_df = pd.read_csv('/mnt/md0/birdclef23/specs/valid.csv')
    
    train_augs = A.ReplayCompose([
        #A.Resize(CFG.image_size[0], CFG.image_size[1]),
        A.RandomResizedCrop(CFG.image_size[0], CFG.image_size[1], scale=[0.2, 1.8], ratio=[0.2, 1.8]),
        #A.Perspective(p=0.5),
        #A.Affine(p=0.5),
        #A.ShiftScaleRotate(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(p=0.5, limit=(-45, 45)),
        #A.RandomBrightnessContrast(p=0.25),
        #A.RandomGamma(p=0.25),
        
        A.CoarseDropout(p=0.5),
        
        A.Normalize(),
        ToTensorV2(),
    ])
    
    valid_augs = A.Compose([
        A.Resize(CFG.image_size[0], CFG.image_size[1]),
        A.Normalize(),
        ToTensorV2(),
    ])
    
    train_dataset = AbdDataset(train_files, train_augs, 1)
    valid_dataset = AbdDataset(valid_files, valid_augs, 0)
    
    if CFG.DDP and CFG.DDP_INIT_DONE:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset, shuffle=True, drop_last=True)
        train_sampler.set_epoch(CFG.epoch) #needed for shuffling?
        train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, sampler=train_sampler, num_workers=CFG.workers, pin_memory=False, drop_last=True)
        
        valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset=valid_dataset, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, sampler=valid_sampler, shuffle=False, num_workers=CFG.workers, pin_memory=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=CFG.workers, pin_memory=False)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=CFG.workers, pin_memory=False)
    
    
    
    #RUINS VALID-LOADER DDP
    #valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=CFG.workers, pin_memory=False)
    
    
    CFG.steps_per_epoch = math.ceil(len(train_loader) / CFG.acc_steps)
    
    return train_loader, valid_loader#, train_data, valid_data

train_loader, valid_loader = get_loaders()

for d in valid_loader: break

#_, axs = plt.subplots(2, 4, figsize=(24, 12))
_, axs = plt.subplots(1, 4, figsize=(30, 6))
axs = axs.flatten()
for img, ax in zip(range(8), axs):
    try:
        ax.imshow(d['images'][img].numpy()[:3].transpose(1, 2, 0))
    except: pass
    
_, axs = plt.subplots(1, 4, figsize=(30, 6))
axs = axs.flatten()
for img, ax in zip(range(8), axs):
    try:
        ax.imshow(d['masks'][img].numpy().mean(0))
    except: pass


# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


import sys
sys.path.append(f'/mnt/md0/dacl10k/coat/src_contrails/')

from collections import OrderedDict
from src.coat import CoaT,coat_lite_mini,coat_lite_small,coat_lite_medium
from src.layers import *

class Model(nn.Module):
    def __init__(self, checkpoint_file=1):
        super(Model, self).__init__()
        
        '''
        from mmseg.apis import inference_segmentor, init_segmentor
        config_file = '/mnt/md0/dacl10k/AAA_MMSEG/mmsegmentation/work_dirs/convnext_large_exp_002/test_config_def.py'
        
        if checkpoint_file: checkpoint_file = '/mnt/md0/dacl10k/AAA_MMSEG/mmsegmentation/work_dirs/convnext_large_exp_002/best_mIoU_iter_45045.pth'
        else: checkpoint_file = None
        #checkpoint_file = None
        
        model = init_segmentor(config_file, checkpoint_file, device='cpu')
        '''
        
        import sys
        sys.path.append("/mnt/md0/dacl10k/AAA_MMSEG/EVA/EVA-02/seg/")
        
        from mmcv.utils import Config
        from mmseg.apis import init_segmentor
        from mmseg.models import build_segmentor
        from backbone import eva2

        config_file = '/mnt/md0/dacl10k/AAA_MMSEG/EVA/EVA-02/seg/work_dirs/eva_02_large_exp_001/upernet_eva02_large_24_512_slide_80k.py'
        if checkpoint_file:
            checkpoint_file = '/mnt/md0/dacl10k/AAA_MMSEG/EVA/EVA-02/seg/work_dirs/eva_02_large_exp_001/best_mIoU_iter_42735.pth'
            st = torch.load(checkpoint_file, map_location='cpu')
            PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                           [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                           [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                           [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                           [143, 255, 140], [204, 255, 4], [255, 51, 7], (0, 0, 0)]
            st['meta']['PALETTE'] = PALETTE
            new_checkpoint_file = checkpoint_file.replace('.pth', '_mod.pth')
            torch.save(st, new_checkpoint_file)
        else:
            new_checkpoint_file = None

        model = init_segmentor(config_file, new_checkpoint_file, device='cpu')
        
        self.backbone = model.backbone
        self.decoder_head = model.decode_head
        
        #self.decoder_head.conv_seg = nn.Conv2d(512, 19, kernel_size=(1, 1), stride=(1, 1))
        
        num_classes = 19
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.head = nn.Linear(1024, num_classes)
        
        '''
        var1 = 1536
        var2 = 768
        var3 = 1280
        var4 = 384
        var5 = 640
        var6 = 192
        var7 = 320
        '''
        
        '''
        var1 = 1024
        var2 = 1024
        var3 = 1024
        var4 = 1024
        var5 = 1024
        var6 = 1024
        var7 = 1024
        
        self.dec4 = UnetBlock(var1, var2, var3)
        self.dec3 = UnetBlock(var3, var4, var5)
        self.dec2 = UnetBlock(var5, var6, var7)
        self.fpn = FPN([var1, var3, var5], [192]*3)
        self.final_conv = nn.Sequential(UpBlock(var7+192*3, num_classes, blur=True))
        '''
        
        
    def forward(self, inp):
        features = self.backbone(inp)
        
        #return features
        
        '''
        dec4 = features[-1]
        dec3 = self.dec4(dec4, features[-2])
        dec2 = self.dec3(dec3, features[-3])
        dec1 = self.dec2(dec2, features[-4])
        fpn_out = self.fpn([dec4, dec3, dec2], dec1)
        masks2 = self.final_conv(fpn_out)
        '''
        
        masks = self.decoder_head(features)
        masks = masks[:, :19]
        
        #masks = nn.functional.interpolate(masks, masks2.shape[-2:])
        
        #masks = (masks * 0.5) + (masks2 * 0.5)
        
        feat = features[-1]
        feat = self.avgpool(feat).flatten(1, 3)
        
        logits = self.head(feat)
        
        masks = nn.functional.interpolate(masks, inp.shape[-2:])
        
        return masks, logits


# In[9]:


if type(CFG.model_name)!=str: CFG.model_name = 'resnet18d'
    
#CFG.model_name = 'tf_efficientnetv2_s_in21ft1k'
'''
torch.cuda.set_device(1)
m = Model().cuda()
m.eval()
with torch.no_grad():
    outs = m(d['images'][:2].cuda())
_ = [print(o.shape) for o in outs]
#'''


# In[ ]:





# In[46]:


class_to_idx = {
    'Crack': 0,
    'Wetspot': 1,
    'ExposedRebars': 2,
    'ACrack': 3,
    'Rust': 4,
    'Bearing': 5,
    'Efflorescence': 6,
    'Graffiti': 7,
    'EJoint': 8,
    'Rockpocket': 9,
    'Weathering': 10,
    'Drainage': 11,
    'WConccor': 12,
    'PEquipment': 13,
    'Hollowareas': 14,
    'JTape': 15,
    'Cavity': 16,
    'Spalling': 17,
    'Restformwork': 18
}

idx_to_class = {class_to_idx[k]: k for k in class_to_idx}

def compute_miou(pred, gt, threshold=0.5):
    """
    Compute mean Intersection over Union (mIoU) for a batch of predictions and ground truth masks using PyTorch.
    
    Args:
    - pred (torch.Tensor): Predicted tensor of shape (batch_size, n_classes, height, width).
    - gt (torch.Tensor): Ground truth tensor of shape (batch_size, n_classes, height, width).
    - threshold (float): Threshold to convert predicted probabilities to binary mask.
    
    Returns:
    - float: mIoU value.
    """
    
    # Convert predicted probabilities to binary mask
    pred = (pred > threshold).float()
    
    # Compute intersection and union
    intersection = torch.sum(pred * gt, dim=(0, 2, 3))
    union = torch.sum(pred, dim=(0, 2, 3)) + torch.sum(gt, dim=(0, 2, 3)) - intersection
    
    # Avoid division by zero
    union = torch.clamp(union, min=1e-10)
    
    # Compute IoU for each class and then average
    iou = intersection / union
    miou = torch.mean(iou)
    
    return miou.item()

def calculate_metric(OUTPUTS, TARGETS, threshold=0.5):
    
    OUTPUTS = torch.as_tensor(OUTPUTS)
    TARGETS = torch.as_tensor(TARGETS)
    
    class_metric = {}
    for i in range(OUTPUTS.shape[1]):
        #iou = 1 - smp.losses.JaccardLoss(mode='binary', from_logits=False)(OUTPUTS[:, i]>threshold, TARGETS[:, i]).item()
        iou = compute_miou(OUTPUTS[:, i:i+1], TARGETS[:, i:i+1], threshold)
        class_metric[idx_to_class[i]] = iou
    
    miou = np.mean([class_metric[key] for key in class_metric])
    
    return miou, class_metric

def cutmix(data, masks, labels, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_masks = masks[indices]
    shuffled_labels = labels[indices]

    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()

    image_h, image_w = data.shape[2:]
    cx = torch.randint(0, image_w, (1,)).item()
    cy = torch.randint(0, image_h, (1,)).item()
    
    cut_w = int(image_w * torch.sqrt(1. - torch.tensor(lam)))
    cut_h = int(image_h * torch.sqrt(1. - torch.tensor(lam)))
    
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, image_w)
    y2 = min(cy + cut_h // 2, image_h)
    
    data[:, :, y1:y2, x1:x2] = shuffled_data[:, :, y1:y2, x1:x2]
    masks[:, :, y1:y2, x1:x2] = shuffled_masks[:, :, y1:y2, x1:x2]
    
    lam = 1 - (x2 - x1) * (y2 - y1) / (image_h * image_w)
    
    mixed_labels = lam * labels + (1 - lam) * shuffled_labels
    targets = ((masks, mixed_labels), (shuffled_masks, shuffled_labels), lam)
    
    return data, targets

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        self.jaccard = smp.losses.JaccardLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        
    def _compute_loss(self, outputs, targets):
        #loss1 = self.dice(outputs, targets)  # Uncomment if you want to use the dice loss
        
        #idxs = [9, 12, 1, 16, 18] #['Rockpocket', 'Wconccor', 'Wetspot', 'Cavity', 'Restformwork']
        loss2 = self.jaccard(outputs, targets)
        loss3 = self.bce(outputs, targets)
        
        #loss4 = self.jaccard(outputs[:, idxs], targets[:, idxs]) * (len(idxs) / 19)
        #loss5 = self.bce(outputs[:, idxs], targets[:, idxs]) * (len(idxs) / 19)
        
        loss = loss2 + loss3 #+ loss4 + loss5
        
        return loss
        
    def forward(self, outputs, targets, cls_outputs, labels):
        outputs = outputs.float()
        loss_aux = self.bce(cls_outputs, labels)
        
        if isinstance(targets, tuple) and len(targets) == 3:
            (true_masks, true_labels), (shuffled_masks, shuffled_labels), lam = targets
            
            #print(true_masks.dtype, true_labels.dtype, shuffled_masks.dtype, shuffled_labels.dtype)
            #print(true_masks.shape, true_labels.shape, shuffled_masks.shape, shuffled_labels.shape)
            
            mask_loss = lam * self._compute_loss(outputs, true_masks) + (1. - lam) * self._compute_loss(outputs, shuffled_masks)
            label_loss = lam * self.bce(cls_outputs, true_labels) + (1. - lam) * self.bce(cls_outputs, shuffled_labels)
            
            loss = loss_aux + mask_loss + label_loss
        else:
            targets = targets.float()
            loss = loss_aux + self._compute_loss(outputs, targets)
        
        return loss

import sys
sys.path.append(f"/mnt/md0/dacl10k/over9000/")
from rangerlars import RangerLars

def get_scheduler(optimizer, plot_while_at_it=False):
    if plot_while_at_it:
        m = nn.Linear(2, 1)
        opt = optim.SGD(m.parameters(), lr=CFG.lr)
        get_scheduler(opt)
        sch = get_cosine_schedule_with_warmup(opt, num_training_steps=CFG.steps_per_epoch * CFG.n_epochs * CFG.upscale_steps, num_warmup_steps=CFG.n_warmup_steps)
        
        lrs = []
        for _ in range(int(CFG.steps_per_epoch * CFG.n_epochs)):
            lr = opt.param_groups[0]['lr']
            lrs.append(lr)
            sch.step()
        plt.plot(range(len(lrs)), np.array(lrs)); plt.show()
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=CFG.steps_per_epoch * CFG.n_epochs * CFG.upscale_steps, num_warmup_steps=CFG.n_warmup_steps)
    
    return scheduler

def define_criterion_optimizer_scheduler_scaler(model):
    criterion = CustomLoss()
    
    #optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    optimizer = RangerLars(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    
    scheduler = get_scheduler(optimizer)
    
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.autocast)
    
    return criterion, optimizer, scheduler, scaler


# In[ ]:





# In[9]:


def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0

    if is_main_process(): bar = tqdm(loader, bar_format='{n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}')
    else: bar = loader
    
    for step, data in enumerate(bar):
        step += 1
        
        images = data['images'].cuda()
        targets = data['masks'].cuda()
        labels = data['labels'].cuda()
        
        if np.random.random()<0.5:
            images, targets = cutmix(images, targets, labels, alpha=0.5)
        
        with torch.cuda.amp.autocast(enabled=CFG.autocast):
            logits, cls_logits = model(images)
        
        loss = criterion(logits, targets, cls_logits, labels)
        
        running_loss += (loss - running_loss) * (1 / step)
        
        loss = loss / CFG.acc_steps
        scaler.scale(loss).backward()
        
        if step % CFG.acc_steps == 0 or step == len(bar):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            CFG.global_step += 1
        
        CFG.literal_step += 1
        
        #lr = "{:2e}".format(next(optimizer.param_groups)['lr'])
        lr = "{:2e}".format(optimizer.param_groups[0]['lr'])
        
        if is_main_process():
            bar.set_postfix(loss=running_loss.item(), lr=float(lr), step=CFG.global_step)
        
        #if step==10: break
        
        dist.barrier()
    
    if is_main_process():
        torch.save(model.module.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}-{CFG.epoch}.pth")
        torch.save(model.module.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth")
        
        
def valid_one_epoch(path, loader, running_dist=True, debug=False):
    model = Model(checkpoint_file=0)
    st = torch.load(path, map_location=f"cpu")
    model.eval()
    model.cuda()
    model.load_state_dict(st, strict=False)
    
    if is_main_process(): bar = tqdm(loader, bar_format='{n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}')
    else: bar = loader
    
    running_loss = 0.
    
    OUTPUTS = {_:[] for _ in range(19)}
    TARGETS = {_:[] for _ in range(19)}
    MASKS_OUTPUTS = []
    MASKS_TARGETS = []
    IDS = []
    mious = []
    
    for step, data in enumerate(bar):
        with torch.no_grad():
            images = data['images'].cuda()
            targets = data['masks'].cuda()
            ids = data['ids']
            
            with torch.cuda.amp.autocast(enabled=CFG.autocast):
                logits, _ = model(images)
                
                #logits2 = model(torch.flip(images, (3,)))
                #logits2 = torch.flip(logits2, (3,))
            
            #logits = logits[:, :, :9]
            #targets = targets[:, :, :9]
            
            outputs = logits.float().sigmoid().detach().cpu().numpy()
            #outputs2 = logits2.float().sigmoid().detach().cpu()#.numpy()
            targets = targets.float().detach().cpu().numpy()
            
            #outputs = (outputs + outputs2) / 2
            
            #ids = np.array(ids)
            
            #'''
            
            if running_dist:
                dist.barrier()
            
                np.save(f'{CFG.cache_dir}/preds_{get_rank()}.npy', outputs)
                np.save(f'{CFG.cache_dir}/targets_{get_rank()}.npy', targets)
                np.save(f'{CFG.cache_dir}/ids_{get_rank()}.npy', ids)

                dist.barrier()
                
                if is_main_process():
                    outputs = np.concatenate([np.load(f"{CFG.cache_dir}/preds_{_}.npy") for _ in range(CFG.N_GPUS)])
                    targets = np.concatenate([np.load(f"{CFG.cache_dir}/targets_{_}.npy") for _ in range(CFG.N_GPUS)])
                    ids = np.concatenate([np.load(f"{CFG.cache_dir}/ids_{_}.npy") for _ in range(CFG.N_GPUS)])
                    
                    miou, class_metric = calculate_metric(outputs, targets)
                    mious.append(miou)
                    
                dist.barrier()
            else:
                for i in range(19):
                    OUTPUTS[i].extend(outputs[:, i]>0.5)
                    TARGETS[i].extend(targets[:, i]>0.5)
                    
                #outputs = np.concatenate([np.zeros((outputs.shape[0], 1, outputs.shape[2], outputs.shape[3])), outputs], 1)
                #targets = np.concatenate([np.zeros((targets.shape[0], 1, targets.shape[2], targets.shape[3])), targets], 1)
                
                #OUTPUTS.extend(outputs_)
                #TARGETS.extend(targets_)
                IDS.extend(ids)
            
            #'''
            
            #OUTPUTS.extend(outputs)
            #TARGETS.extend(targets)
            #IDS.extend(ids)
        
        #running_loss += loss.item()
        
        #if step==5: break
    
    #if len(OUTPUTS):
    #    OUTPUTS = np.stack(OUTPUTS)#.cuda()
    #    TARGETS = np.stack(TARGETS)#.cuda()
        #IDS = torch.cat(IDS).cuda()
    #    IDS = np.stack(IDS)
    
    #OUTPUTS = sync_across_gpus(OUTPUTS, local_rank, world_size).cpu().numpy()
    #TARGETS = sync_across_gpus(TARGETS, local_rank, world_size).cpu().numpy()
    #IDS = sync_across_gpus(IDS, local_rank, world_size).cpu().numpy()
    
    #print(OUTPUTS.shape, TARGETS.shape)
    
    #OUTPUTS = OUTPUTS[:, :, :9]
    #TARGETS = TARGETS[:, :, :9]
    #from mmseg.core.evaluation import mean_iou
    #miou = mean_iou(OUTPUTS, TARGETS, num_classes=1+19, ignore_index=None)
    #print(miou['IoU'][1:].mean())
    
    if running_dist:
        dist.barrier()
    
    #if is_main_process():
        #np.save(f'{OUTPUT_FOLDER}/OUTPUTS_{CFG.FOLD}_last.npy', OUTPUTS)
        #np.save(f'{OUTPUT_FOLDER}/TARGETS_{CFG.FOLD}_last.npy', TARGETS)
        #np.save(f'{OUTPUT_FOLDER}/MASKS_OUTPUTS.npy', np.array(MASKS_OUTPUTS))
        #np.save(f'{OUTPUT_FOLDER}/MASKS_TARGETS.npy', np.array(MASKS_TARGETS))
        #np.save(f'{OUTPUT_FOLDER}/IDS_{CFG.FOLD}_last.npy', IDS)
    
    if running_dist:
        dist.barrier()
    
    if is_main_process():
        MIOUS = np.array(mious)
        miou = np.mean(MIOUS)
    
        print(f"EPOCH {CFG.epoch+1} | mIOU {miou}")
        #print(class_metric)
    
        if debug:
            return miou, OUTPUTS, TARGETS, IDS
    
        return miou
    
    return -1

def run(model, get_loaders):
    if is_main_process():
        epochs = []
        scores = []
    
    best_score = float('-inf')
    for epoch in range(CFG.n_epochs):
        CFG.epoch = epoch
        
        train_loader, valid_loader = get_loaders()
        
        train_one_epoch(model, train_loader)
        
        dist.barrier()
        
        if (CFG.epoch+1)%CFG.validate_every==0 or epoch==0:
            score = valid_one_epoch(f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth", valid_loader, debug=False, running_dist=True)
        
        dist.barrier()
        
        if is_main_process():
            epochs.append(epoch)
            scores.append(score)
            
            if score > best_score:
                print("SAVING BEST!")
                torch.save(model.module.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}_best.pth")
                best_score = score
                
                #np.save(f'{OUTPUT_FOLDER}/OUTPUTS_{CFG.FOLD}_best.npy', OUTPUTS)
                #np.save(f'{OUTPUT_FOLDER}/TARGETS_{CFG.FOLD}_best.npy', TARGETS)
                #np.save(f'{OUTPUT_FOLDER}/MASKS_OUTPUTS.npy', np.array(MASKS_OUTPUTS))
                #np.save(f'{OUTPUT_FOLDER}/MASKS_TARGETS.npy', np.array(MASKS_TARGETS))
                #np.save(f'{OUTPUT_FOLDER}/IDS_{CFG.FOLD}_best.npy', IDS)
            
            try:
                command.run(['rm', '-r', CFG.cache_dir])
                pass
            except:
                pass
            
            os.makedirs(CFG.cache_dir, exist_ok=1)


# In[ ]:





# In[42]:


'''
torch.cuda.set_device(1)

CFG.image_size = (512, 512)
train_loader, valid_loader = get_loaders()
#miou = valid_one_epoch(f"/mnt/md0/dacl10k/AAA_SEG/TRY1_SEG/coat_lite_medium_v4/0_best.pth", valid_loader, debug=False, running_dist=False)
#miou, outputs, targets, ids = valid_one_epoch(f"/mnt/md0/dacl10k/AAA_MMSEG/TRY1_SEG/convnext_large_exp_002_forward_v6/0_best.pth", valid_loader, debug=True, running_dist=False)
miou, outputs, targets, ids = valid_one_epoch(f"/mnt/md0/dacl10k/AAA_MMSEG/TRY1_SEG/eva_02_large_exp_001_v3/0_best.pth", valid_loader, debug=True, running_dist=False)

from mmseg.core.evaluation import mean_iou
ious = []
for i in tqdm(range(19)):
    iou = mean_iou(np.stack(outputs[i]), np.stack(targets[i]), num_classes=2, ignore_index=None)
    iou = iou['IoU'][1]
    ious.append(iou)
    #break
    
miou = np.mean(ious)
classwise = {idx_to_class[i]: v for i, v in enumerate(ious)}

#convnext_large_v2 -> 0.4348057685618855
#v5 (640x640) -> 0.43756302751846804 -> LB 0.455
#v6 (640x640) -> 0.44192072058937776 
#v6 (512x512) -> 0.44020324686364115 -> LB 0.454
#eca v1 (512x512) e7 -> 0.45057784471179047 -> LB 0.4683
#eca v1 (512x512) e11 -> 0.4521060549505043 -> LB 0.459
#eca v2 (512x512) e4 -> 0.4497079883949747
#eca v2 (512x512) e6 -> 0.45919728105742874 -> LB 0.4678
#eca v2 (512x512) e8 -> 0.4561751030991198 -> LB 0.4672
#eca v2 (512x512) e18 -> 0.46113500290017373
#eca v2 (512x512) e21 -> 0.46120072589090577 -> LB 0.472
#eca v2 (512x512) e22 -> 0.4642019633866175 -> LB 0.472
#eca v2 (512x512) e29 -> 0.46324754451359623 -> LB 0.475
#eca v2 (512x512) -> best batch-wise -> LB 0.469

#eca v3 (512x512) -> 0.46358918329215754 -> LB 0.469

#'''


# In[ ]:





# In[ ]:


CFG.DDP = 1

if __name__ == '__main__' and CFG.DDP:
    
    world_size = init_distributed()
    CFG.DDP_INIT_DONE = 1
    
    #important to setup before defining scheduler to establish the correct number of steps per epoch
    train_loader, valid_loader = get_loaders()
    
    model = Model().cuda()
    
    #'''
    st = torch.load(f"/mnt/md0/dacl10k/AAA_MMSEG/TRY1_SEG/eva_02_large_exp_001_v4/{CFG.FOLD}_best.pth", map_location='cpu')
    
    #keys = list(st.keys())
    #for key in keys:
    #    if 'head' in key:
    #        st.pop(key)
    
    model.load_state_dict(st, strict=False)
    #'''
    
    #for param in model.encoder.parameters():
    #    param.requires_grad = False
    
    #model.load_state_dict(torch.load(f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth", map_location='cpu'), strict=False)
    
    if is_main_process():
        torch.save(model.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth")
        
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    local_rank = int(os.environ['LOCAL_RANK'])
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    
    criterion, optimizer, scheduler, scaler = define_criterion_optimizer_scheduler_scaler(model)
    
    run(model, get_loaders)
    
else:
    #important to setup before defining scheduler to establish the correct number of steps per epoch
    train_loader, valid_loader = get_loaders()
    
    model = convert_3d(Model()).cuda()
    
    #model.load_state_dict(torch.load(f"/mnt/md0/rsna_spine/AAA_CLS/TRY12_CLS/b5_v1/best_f{CFG.FOLD}.pth", map_location='cpu'), strict=False)
    
    criterion, optimizer, scheduler, scaler = define_criterion_optimizer_scheduler_scaler(model)
    
    run(model, get_loaders)
    
import sys
sys.exit(0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




