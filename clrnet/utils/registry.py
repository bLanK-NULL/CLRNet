import inspect

import six

# borrow from mmdetection


def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)

"""
创建一个注册表, 动态注册和管理神经网络模块
通过字符串名称来实例化模块, 从而实现模块的灵活配置
"""
class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict() # 字典, 存储所有模块

    #返回注册表的字符串表示，便于调试和日志记
    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    # name 和 module_dict 属性 提供对注册表名称和模块字典的只读访问。
    @property
    def name(self):
        return self._name
    @property
    def module_dict(self):
        return self._module_dict
    # 方法 - 根据模块名称（key）获取已注册的模块类。
    def get(self, key):
        return self._module_dict.get(key, None)
    # 注册一个模块类
    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls

# 根据cfg 动态的构建一个模块
def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy() # args 是 cfg 的副本，用于保存剩余的配置参数。
    print('args:', args)
    obj_type = args.pop('type') #模块类型
    if is_str(obj_type):
        obj_cls = registry.get(obj_type) # 从注册表中取得模块
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)
