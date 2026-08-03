import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import random


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


def subbands_mixup(sub_s, sub_t, th):
    sub_s = sub_s.to(torch.device("cuda"))
    sub_s[0, 3:6, :, :] = sub_s[0, 3:6, :, :] * th + sub_t[0, 3:6, :, :] * (1-th)
    sub_s[0, 6:9, :, :] = sub_s[0, 6:9, :, :] * th + sub_t[0, 6:9, :, :] * (1-th)
    sub_s[0, 9:12, :, :] = sub_s[0, 9:12, :, :] * th + sub_t[0, 9:12, :, :] * (1 - th)

    return sub_s


def source_mixup(x_s, target_s, mix_t, th):
    for i, img in enumerate(x_s):
        img = torch.unsqueeze(img, 0)
        sub_s = DWT()(img)
        img_t = random.sample(mix_t[target_s[i]], 1)[0]
        img_t = torch.unsqueeze(img_t, 0)
        sub_t = DWT()(img_t).to(torch.device("cuda"))
        sub_s = subbands_mixup(sub_s, sub_t, th)
        reconstruction_img = IWT()(sub_s)
        x_s[i] = reconstruction_img[0]

    return x_s


def mix_t_initial(x_t, target_t, s2m):
    mix_t = [[],[],[],[],[],[],[],[],[],[]]
    if s2m:
        mix_t = [[], [], [], [], []]
    for i, img_t in enumerate(x_t):
        mix_t[target_t[i]].append(img_t)

    return mix_t


def mix_t_append(mix_t, un_img_list, un_target_list):
    if len(un_img_list) != 0:
        for i, un_img in enumerate(un_img_list):
            mix_t[un_target_list[i]].append(un_img)
    return mix_t


def mix_feature_initial(x_feature, target_t, s2m, device):
    mix_feature = []
    labels = []
    for i, img_feature in enumerate(x_feature):
        mix_feature.append(img_feature)
        labels.append(target_t[i])
    mix_feature_tensor = torch.stack(mix_feature, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device) 
    return mix_feature_tensor, labels_tensor


def mix_feature_append(mix_feature_tensor, labels_tensor, un_img_feature, un_target_list, device):
    if len(un_img_feature) != 0:
        new_features = torch.stack(un_img_feature, dim=0)
        new_labels = torch.tensor(un_target_list, dtype=torch.long, device=device) 
        mix_feature_tensor = torch.cat((mix_feature_tensor, new_features), dim=0)
        labels_tensor = torch.cat((labels_tensor, new_labels), dim=0)
    return mix_feature_tensor, labels_tensor

def proto_count(mix_feature_tensor, labels, num_classes, feature_dim):
    proto_tensor = torch.zeros((num_classes, feature_dim), device=mix_feature_tensor.device)
    for i in range(num_classes):
        class_features = mix_feature_tensor[labels == i]
        if len(class_features) > 0:
            proto_ins = class_features.mean(dim=0)
        else:
            proto_ins = torch.zeros(feature_dim, device=mix_feature_tensor.device)
        proto_tensor[i] = proto_ins

    return proto_tensor
