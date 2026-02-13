# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class SaveSelectedEpochsHook(Hook):
    """Save checkpoints only at specified epochs.

    Args:
        epochs (list[int]): 1-based epoch numbers to save.
        save_optimizer (bool): Whether to save optimizer state.
        save_last (bool): Whether to also save the last epoch.
        filename_tmpl (str): Filename template used by runner.save_checkpoint.
    """

    def __init__(self,
                 epochs,
                 save_optimizer=True,
                 save_last=False,
                 filename_tmpl='epoch_{}.pth'):
        self.epochs = set(int(e) for e in epochs)
        self.save_optimizer = save_optimizer
        self.save_last = save_last
        self.filename_tmpl = filename_tmpl

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        if epoch in self.epochs:
            runner.save_checkpoint(
                runner.work_dir,
                filename_tmpl=self.filename_tmpl,
                save_optimizer=self.save_optimizer,
                meta=dict(epoch=epoch))

    def after_run(self, runner):
        if self.save_last:
            epoch = runner.epoch + 1
            runner.save_checkpoint(
                runner.work_dir,
                filename_tmpl=self.filename_tmpl,
                save_optimizer=self.save_optimizer,
                meta=dict(epoch=epoch))
