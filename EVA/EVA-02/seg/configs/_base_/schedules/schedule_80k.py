# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
bs = 2
runner = dict(type='IterBasedRunner', max_iters=int(6935/(bs * 3)) * 40)
checkpoint_config = dict(interval=int(6935/(bs * 3)), max_keep_ckpts=5)
evaluation = dict(interval=int(6935/(bs * 3)), metric='mIoU', pre_eval=True, save_best='mIoU')
