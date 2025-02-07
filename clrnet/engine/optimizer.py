import torch

# 构建优化器
'''
调整模型的参数（如权重和偏置）来最小化损失函数
'''
def build_optimizer(cfg, net):
    params = []
    cfg_cp = cfg.optimizer.copy()
    cfg_type = cfg_cp.pop('type')

    if cfg_type not in dir(torch.optim):
        raise ValueError("{} is not defined.".format(cfg_type))

    _optim = getattr(torch.optim, cfg_type)
    return _optim(net.parameters(), **cfg_cp)
