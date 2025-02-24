import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os

from clrnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from clrnet.datasets import build_dataloader
from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel


class Runner(object):
    def __init__(self, cfg):
        # 设置 PyTorch 和 numpy  的随机种子，确保实验的可重复性
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        # 保存配置
        self.cfg = cfg
        # 构建记录器（Recorder），用于记录训练过程中的日志、指标等信息
        self.recorder = build_recorder(self.cfg)
        # 构建网络模型，根据配置文件中的模型架构初始化模型
        self.net = build_net(self.cfg)
        # 将模型包装为多 GPU 并行模型（MMDataParallel），并移动到 GPU 上
        # device_ids 指定使用的 GPU 设备范围，self.cfg.gpus 是 GPU 数量
        self.net = MMDataParallel(self.net, device_ids=range(self.cfg.gpus)).cuda()
        # 使用记录器记录网络模型的架构信息
        self.recorder.logger.info('Network: \n' + str(self.net))
        # 恢复训练（如果指定了 resume_from 或 load_from 参数）
        # 从检查点加载模型权重和优化器状态
        self.resume()
        # 构建优化器（Optimizer），根据配置文件中的优化器参数初始化优化器
        self.optimizer = build_optimizer(self.cfg, self.net)
        # 构建学习率调度器（Scheduler），根据配置文件中的调度器参数初始化调度器
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        # 初始化一个变量 self.metric，用于保存评估指标（如精度、损失等）
        self.metric = 0.
        # 初始化验证集数据加载器为 None，后续根据需要构建
        self.val_loader = None
        # 初始化测试集数据加载器为 None，后续根据需要构建
        self.test_loader = None

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch
    # 用来从断点继续 需要命令行参数 --resume_from
    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader):
        self.net.train() # 模型模式设置为训练模式
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data) # 数据移动到gpu
            output = self.net(data) # 前向传播, 数据输入模型
            self.optimizer.zero_grad()
            loss = output['loss'].sum() # 提取损失值，并对多个损失（如果有）求和。
            loss.backward() # 反向传播 计算梯度
            self.optimizer.step() #使用优化器更新模型参数
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step() # 跟新学习率
            batch_time = time.time() - end # 计算一个 batch 的时间
            end = time.time()
            # 记录
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            # 每隔 log_interval 次迭代或在最后一个迭代时，记录当前的学习率和训练统计信息。
            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Build train loader...')
        # 加载训练集
        train_loader = build_dataloader(self.cfg.dataset.train,
                                        self.cfg,
                                        is_train=True)

        self.recorder.logger.info('Start training...')
        start_epoch = 0
        if self.cfg.resume_from:
            start_epoch = resume_network(self.cfg.resume_from, self.net,
                                         self.optimizer, self.scheduler,
                                         self.recorder)
        # 循环训练，每个 epoch 训练完后，进行验证和保存模型
        for epoch in range(start_epoch, self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch +
                    1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt() # 保存当前模型的检查点
            if (epoch +
                    1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate() # 定期验证, 监控模型的性能、防止过拟合
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch: # 各个配置都是False
                self.scheduler.step()


    def test(self):
        if not self.test_loader:
            self.test_loader = build_dataloader(self.cfg.dataset.test,
                                                self.cfg,
                                                is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.test_loader.dataset.view(output, data['meta'])

        metric = self.test_loader.dataset.evaluate(predictions,
                                                   self.cfg.work_dir)
        if metric is not None:
            self.recorder.logger.info('metric: ' + str(metric))

    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val,
                                               self.cfg,
                                               is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        metric = self.val_loader.dataset.evaluate(predictions,
                                                  self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder,
                   is_best)
