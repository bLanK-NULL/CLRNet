net = dict(type='Detector', )

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True, # 加载与训练权重
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

num_points = 72
max_lanes = 5
sample_y = range(710, 150, -10)

heads = dict(type='CLRHead',
             num_priors=192, # 每个像素点的先验框数量
             refine_layers=3, # 精调层的数量
             fc_hidden_dim=64,# 全连接层的隐藏单元数量
             sample_points=36) # 采样点数量

iou_loss_weight = 2. # iou损失权重
cls_loss_weight = 6. # 分类损失权重
xyt_loss_weight = 0.5 # xyt损失权重
seg_loss_weight = 1.0 # 分割损失权重

work_dirs = "work_dirs/clr/r18_tusimple"
# 定义模型的颈部（neck）为 FPN（特征金字塔网络），用于融合不同尺度的特征
neck = dict(type='FPN',
            in_channels=[128, 256, 512], #输入特征图的通道数
            out_channels=64, #输出特征图的通道数
            num_outs=3, #输出特征图的数量
            attention=False) #是否使用注意力机制

test_parameters = dict(conf_threshold=0.40, nms_thres=50, nms_topk=max_lanes)

epochs = 70
batch_size = 40

optimizer = dict(type='AdamW', lr=1.0e-3)  # 3e-4 for batchsize 8
total_iter = (3616 // batch_size + 1) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_ep = 3
save_ep = epochs

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w = 1280
ori_img_h = 720
img_h = 320
img_w = 800
cut_height = 160 

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = './data/tusimple'
dataset_type = 'TuSimple'
test_json_file = 'data/tusimple/test_label.json'
dataset = dict(train=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='trainval',
    processes=train_process,
),
val=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
),
test=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
))

workers = 10
log_interval = 100
# seed = 0
num_classes = 6 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
