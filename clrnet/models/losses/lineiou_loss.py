import torch


def line_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, num_pred是真实车道线的数量，72 是每个真实车道线的特征数量
        img_w: image width
        length: 车道线宽度 默认为15
        aligned: True for iou loss calculation, False for pair-wise ious in assign
        如果 aligned 为 True，则计算每个预测和对应的真实车道线之间的交集和并集。
        如果 aligned 为 False，则计算每个预测和所有真实车道线之间的交集和并集。
    Returns:
        shape: (num_pred)
    '''
    px1 = pred - length  # 预测车道线左边界
    px2 = pred + length # 预测车道线右边界
    tx1 = target - length
    tx2 = target + length
    if aligned:
        '''
        invalid_mask 用于标记无效区域，即车道线超出图像宽度的部分。
        对于无效区域，交集和并集的值设置为 0。
        '''
        invalid_mask = target
        # ovr 交集
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        # union 并集
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9) # + 1e-9 是为了防止除以0的发生
    return iou


def liou_loss(pred, target, img_w, length=15):
    return (1 - line_iou(pred, target, img_w, length)).mean()