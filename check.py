import torch
import torchvision

# 查看 PyTorch 版本
print("PyTorch 版本:", torch.__version__)

# 查看 CUDA 版本
print("CUDA 版本:", torch.version.cuda)

# 检查 CUDA 是否可用
print("CUDA 是否可用:", torch.cuda.is_available())

# 查看当前使用的 GPU 设备信息
if torch.cuda.is_available():
    print("当前 GPU 设备:", torch.cuda.get_device_name(0))
else:
    print("没有可用的 GPU 设备")

# 查看 torchvision 版本
print("torchvision 版本:", torchvision.__version__)
