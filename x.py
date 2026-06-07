# -*- coding: utf-8 -*-
"""
TensorBoard Hook for MMDiT Training
=====================================
记录内容：
  标量（每 scalar_interval 步）
    train/loss              当前步 flow matching loss
    train/grad_norm         裁剪前梯度范数
    train/grad_norm_clipped 裁剪后梯度范数
    train/lr                当前学习率

  图像（每 image_interval 步，需要 trainer 存 _tb_batch，见下）
    train/cond_frame        首帧条件图（VAE latent 解码为像素图）

启用：
  1. trainer_mmdit.py run_step() 末尾加一行：
         self._tb_batch = batch_data

  2. engine/__init__.py 的 hook_list 里加：
         TensorBoardHook(cfg)

查看：
  tensorboard --logdir logs/<yml_name>/tensorboard
"""

import os
import threading
import torch
import torch.distributed as dist
from .train_loop import HookBase
from mimogpt.utils import hf_logger

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


class TensorBoardHook(HookBase):

    def __init__(self, cfg, scalar_interval=None, image_interval=500, max_images=2, upload_interval=None):
        """
        scalar_interval : 记录标量的步数间隔（None = 用 cfg.common.log_interval）
        image_interval  : 记录图像的步数间隔（0 = 不记录图像）
        max_images      : 每次最多记录几张图，防止占用太多显存
        upload_interval : 上传到 mox 的步数间隔（None = 用 save_per_steps；0 = 不上传）
        """
        self.cfg = cfg
        self.is_root = (dist.get_rank() == 0)
        self.scalar_interval = scalar_interval or cfg.common.log_interval
        self.image_interval = image_interval
        self.max_images = max_images

        # 上传间隔：默认和保存模型同频
        if upload_interval is None:
            self.upload_interval = getattr(cfg.common, 'save_per_steps', 500)
        else:
            self.upload_interval = upload_interval

        # mox 上传目标路径
        self.train_url = getattr(cfg, 'train_url', None)

        if self.is_root:
            if not HAS_TB:
                hf_logger.warning("TensorBoardHook: tensorboard not installed, hook disabled")
                self.writer = None
                self.log_dir = None
            else:
                self.log_dir = os.path.join(cfg.common.log_path, 'tensorboard')
                os.makedirs(self.log_dir, exist_ok=True)
                self.writer = SummaryWriter(self.log_dir, max_queue=10, flush_secs=30)
                hf_logger.info(f"TensorBoardHook: writing to {self.log_dir}")
        else:
            self.writer = None
            self.log_dir = None

    # ──────────────────────────────────────────────────────────
    # 生命周期
    # ──────────────────────────────────────────────────────────

    def after_step(self):
        if self.writer is None:
            return

        step = self.trainer.iter

        if (step + 1) % self.scalar_interval == 0:
            self._log_scalars(step)

        if self.image_interval > 0 and (step + 1) % self.image_interval == 0:
            self._log_images(step)

        if self.upload_interval > 0 and (step + 1) % self.upload_interval == 0:
            self.writer.flush()
            self._upload_to_mox()

    def after_train(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        self._upload_to_mox()

    # ──────────────────────────────────────────────────────────
    # mox 上传
    # ──────────────────────────────────────────────────────────

    def _upload_to_mox(self):
        """把本地 tensorboard 目录异步上传到 mox（train_url/tensorboard）。"""
        if not self.is_root:
            return
        if self.log_dir is None:
            return
        if not self.train_url:
            return

        dst = os.path.join(self.train_url, 'tensorboard')

        def _do_upload():
            try:
                from mimogpt.engine.utils.cloud_copy import mox_copy
                mox_copy(self.log_dir, dst, parallel=True)
                hf_logger.info(f"TensorBoardHook: uploaded to {dst}")
            except Exception as e:
                hf_logger.warning(f"TensorBoardHook: mox upload failed: {e}")

        t = threading.Thread(target=_do_upload, daemon=True)
        t.start()

    # ──────────────────────────────────────────────────────────
    # 标量
    # ──────────────────────────────────────────────────────────

    def _log_scalars(self, step):
        # _loss 在 run_step 里已经除以了 accumulation_steps，乘回来得到真实 loss
        real_loss = self.trainer._loss.item() * self.cfg.common.gradient_accumulation_steps
        self.writer.add_scalar('train/loss', real_loss, step)

        if hasattr(self.trainer, 'grad_norm'):
            self.writer.add_scalar('train/grad_norm',
                                   float(self.trainer.grad_norm), step)
        if hasattr(self.trainer, 'grad_norm_clipped'):
            self.writer.add_scalar('train/grad_norm_clipped',
                                   float(self.trainer.grad_norm_clipped), step)

        lr = self.trainer.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/lr', lr, step)

    # ──────────────────────────────────────────────────────────
    # 图像
    # ──────────────────────────────────────────────────────────

    def _log_images(self, step):
        """
        把首帧 VAE latent 解码成像素图写入 TensorBoard。

        依赖：trainer_mmdit.py 的 run_step() 末尾需要加：
            self._tb_batch = batch_data
        """
        batch = getattr(self.trainer, '_tb_batch', None)
        if batch is None:
            return

        cond_latent = batch.get('image', None)
        if cond_latent is None:
            return

        vae = getattr(self.trainer, 'vae', None)
        if vae is None:
            # online_vae=False 时 trainer 没有 vae，跳过
            return

        try:
            B = min(cond_latent.shape[0], self.max_images)
            z = cond_latent[:B].cuda().to(torch.bfloat16)  # [B, 16, 1, H_lat, W_lat]

            with torch.no_grad():
                # MotionVAE_Wrapper.decode(z) → [B, 3, T, H, W]
                decoded = vae.decode(z)

            # 取第一帧，转为 float，归一化到 [0, 1]
            frame = decoded[:, :, 0].float().clamp(-1.0, 1.0)  # [B, 3, H, W]
            frame = (frame + 1.0) / 2.0
            self.writer.add_images('train/cond_frame', frame.cpu(), step)

        except Exception as e:
            hf_logger.warning(f"TensorBoardHook: image logging failed: {e}")
