import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from utils import kernel
import os
from openpyxl import Workbook


def get_losses_unlabeled(args, G, F1, im_data, im_data_bar2, targets_u, im_data_t, gt_labels_t, group_consistency, step, enhanced=False, mix_add=False, proto_add=False):
    feat = G(im_data)
    feat_bar2 = G(im_data_bar2)

    output = F1(feat) 
    output_bar2 = F1(feat_bar2) 

    prob = F.softmax(output, dim=1) 

    if group_consistency:
        feature_map_x = G(im_data_t)
        label_uw = torch.cat((feat, feature_map_x), dim=0)
        label_us = torch.cat((feat_bar2, feature_map_x), dim=0)
        target = torch.cat((targets_u, gt_labels_t), dim=0)
        sigma = 0.5
        ker = kernel.kernelTrans(label_uw, target, ['rbf', sigma])
        ker2 = kernel.kernelTrans(label_us, target, ['rbf', sigma])
        group_loss = F.mse_loss(ker, ker2, size_average=False) / (len(target))
    else:
        group_loss = torch.tensor(0, dtype=float)


    max_probs, pseudo_labels = torch.max(prob.detach_(), dim=-1)
    mask = max_probs.ge(args.threshold).float()


    un_img_list = []
    un_img_feature = []
    un_target_list = []
    
    for i, index in enumerate(mask):
        if index == 1:
            un_img_list.append(im_data[i])
            un_target_list.append(pseudo_labels[i])
            un_img_feature.append(feat[i])
    pl_loss = (F.cross_entropy(output_bar2, pseudo_labels, reduction='none') * mask).mean()
    return pl_loss, group_loss, un_img_list, un_img_feature, un_target_list

def save_similarity(ker1, ker2, step):

    ker1 = ker1.tolist()
    ker2 = ker2.tolist()
    mybook = Workbook()
    wa1 = mybook.active
    matric_save_path1 = os.path.join(os.getcwd(), "similarity_matirc_sample", str(step) + "_" + 'ker1.xlsx')
    for i in range(len(ker1)):
        wa1.append(ker1[i])
    mybook.save(matric_save_path1)

    wa2 = mybook.active
    for i in range(len(ker2)):
        wa2.append(ker2[i])
    matric_save_path2 = os.path.join(os.getcwd(), "similarity_matirc_sample", str(step) + "_" + 'ker2.xlsx')
    mybook.save(matric_save_path2)

class PrototypeLoss(object):
    def __init__(self, ways=10, trg_shots=3, src_shots=10):
        self.ways = ways
        self.trg_shots = trg_shots
        self.src_shots = src_shots

        label = torch.arange(self.ways).unsqueeze(0).repeat(self.src_shots, 1).t().contiguous().view(-1).squeeze(0)
        label = label.type(torch.cuda.LongTensor)
        self.label = label


    def __call__(self, src_feat, proto, gt_labels_s, normalize_feature=False):
        n = src_feat.shape[0]
        m = proto.shape[0]
        src_feat = src_feat.unsqueeze(1).expand(n, m, -1)   
        proto = proto.unsqueeze(0).expand(n, m, -1) 
        logits = -torch.norm(src_feat - proto, 2, -1, keepdim=False)
        loss = F.cross_entropy(logits, gt_labels_s)

        return loss

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def proto_count(mix_t, G):
    proto = []
    for _, mix_ins in enumerate(mix_t):
        mix_ins = torch.stack(mix_ins)
        mix_feature = G(mix_ins)
        proto_ins = mix_feature.mean(dim=0)
        proto.append(proto_ins)
    proto = torch.stack(proto)

    return proto

def advbce_unlabeled(args, target, feat, prob, prob_bar, device, bce):
    """ Construct adversarial adpative clustering loss."""
    target_ulb = pairwise_target(args, feat, target, device) 
    prob_bottleneck_row, _ = PairEnum2D(prob)   
    _, prob_bottleneck_col = PairEnum2D(prob_bar)  
    adv_bce_loss = -bce(prob_bottleneck_row, prob_bottleneck_col, target_ulb)
    return adv_bce_loss

def pairwise_target(args, feat, target, device):
    """ Produce pairwise similarity label."""
    feat_detach = feat.detach()
    if target is None:
        rank_feat = feat_detach
        rank_idx = torch.argsort(rank_feat, dim=1, descending=True) 
        rank_idx1, rank_idx2 = PairEnum2D(rank_idx)
        rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
        rank_idx1, _ = torch.sort(rank_idx1, dim=1)
        rank_idx2, _ = torch.sort(rank_idx2, dim=1)
        rank_diff = rank_idx1 - rank_idx2
        rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
        target_ulb = torch.ones_like(rank_diff).float().to(device)
        target_ulb[rank_diff > 0] = 0
    elif target is not None:
        target_row, target_col = PairEnum1D(target)
        target_ulb = torch.zeros(target.size(0) * target.size(0)).float().to(device)
        target_ulb[target_row == target_col] = 1
    else:
        raise ValueError('Please check your target.')
    return target_ulb

def PairEnum1D(x):
    """ Enumerate all pairs of feature in x with 1 dimension."""
    assert x.ndimension() == 1, 'Input dimension must be 1'
    x1 = x.repeat(x.size(0), )
    x2 = x.repeat(x.size(0)).view(-1, x.size(0)).transpose(1, 0).reshape(-1)
    return x1, x2

def PairEnum2D(x):
    """ Enumerate all pairs of feature in x with 2 dimensions."""
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    return x1, x2

def sigmoid_rampup(current, rampup_length):
    """ Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

class BCE(nn.Module):
    eps = 1e-7
    def forward(self, prob1, prob2, simi):
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

class BCE_softlabels(nn.Module):
    """ Construct binary cross-entropy loss."""
    eps = 1e-7
    def forward(self, prob1, prob2, simi): 
        P = prob1.mul_(prob2) 
        P = P.sum(1)  
        neglogP = - (simi * torch.log(P + BCE.eps) + (1. - simi) * torch.log(1. - P + BCE.eps))
        return neglogP.mean()


def get_loss(pro_t, arche_t, pro_s, arche_s, tau):
    hyper_loss = get_hyper_loss(pro_t, arche_t, pro_s, arche_s, tau)  
    print("hyper_loss:", hyper_loss)
    type_loss = get_type_loss(pro_t, pro_s, tau)
    print("type_loss:", type_loss)
    loss = hyper_loss + type_loss
    return loss


def theta(pro_t, pro_s):
    return torch.norm(pro_s - pro_t, p=2, dim=1)


def get_hyper_loss(pro_t, arche_t, pro_s, arche_s, tau):
    arche_deviation1 = theta(pro_t, arche_t)
    arche_deviation2 = theta(pro_s, arche_s)
    mse_loss = torch.mean((arche_deviation1 - arche_deviation2) ** 2)

    N = 10
    theta_matrix = torch.zeros(N, N)
    for i in range(N):
        theta_matrix[i] = theta(pro_s[i].unsqueeze(0), pro_t)
    numerator_pro = torch.exp(theta_matrix.diag())
    denominator_pro = torch.exp(theta_matrix).sum(dim=1) - numerator_pro
    valid_mask = denominator_pro > 1e-8
    center_loss = -torch.sum(torch.log(numerator_pro[valid_mask] / denominator_pro[valid_mask]))

    radient_loss_list = []
    for i in range(N):
        numerator_distance = torch.norm(torch.cat([pro_s[i], arche_s[i]]) - torch.cat([pro_t[i], arche_t[i]]), p=2)
        numerator_arche = torch.exp(numerator_distance)
        denominator_arche = 0
        for j in range(N):
            if j != i:
                denominator_distance = torch.norm(torch.cat([pro_s[i], arche_s[j]]) - torch.cat([pro_t[j], arche_t[i]]),
                                                  p=2)
                denominator_arche += torch.exp(denominator_distance)
        if denominator_arche > 1e-8:
            radient_loss_list.append(torch.log(numerator_arche / denominator_arche))
    radient_loss = -torch.sum(torch.stack(radient_loss_list))

    print("center_loss:", center_loss)
    print("radient_loss:", radient_loss)
    print("mse_loss:", mse_loss)
    hyper_loss = 0.1 * center_loss + 0.01 * radient_loss + 100 * mse_loss
    return hyper_loss


def get_type_loss(pro_t, pro_s, tau):
    N = 10
    loss_l_his_ht = 0.0
    loss_l_ht_his = 0.0
    for i in range(N):
        theta_his_ht = theta(pro_s[i].unsqueeze(0), pro_t[i].unsqueeze(0)) / tau
        numerator_l_his_ht = torch.exp(theta_his_ht)
        denom1_l_his_ht = 0.0
        denom2_l_his_ht = 0.0
        for k in range(N):
            if k != i:
                denom1_l_his_ht += torch.exp(theta(pro_s[i].unsqueeze(0), pro_t[k].unsqueeze(0)) / tau)
                denom2_l_his_ht += torch.exp(theta(pro_s[i].unsqueeze(0), pro_s[k].unsqueeze(0)) / tau)
        denominator_l_his_ht = denom1_l_his_ht + denom2_l_his_ht
        if denominator_l_his_ht > 1e-8:
            loss_l_his_ht += torch.log(numerator_l_his_ht / denominator_l_his_ht)

        theta_ht_his = theta(pro_t[i].unsqueeze(0), pro_s[i].unsqueeze(0)) / tau
        numerator_l_ht_his = torch.exp(theta_ht_his)
        denom1_l_ht_his = 0.0
        denom2_l_ht_his = 0.0
        for k in range(N):
            if k != i:
                denom1_l_ht_his += torch.exp(theta(pro_t[i].unsqueeze(0), pro_s[k].unsqueeze(0)) / tau)
                denom2_l_ht_his += torch.exp(theta(pro_t[i].unsqueeze(0), pro_t[k].unsqueeze(0)) / tau)
        denominator_l_ht_his = denom1_l_ht_his + denom2_l_ht_his
        if denominator_l_ht_his > 1e-8:
            loss_l_ht_his += torch.log(numerator_l_ht_his / denominator_l_ht_his)

    L_his_ht = -loss_l_his_ht
    L_ht_his = -loss_l_ht_his
    L_prototype = (L_his_ht + L_ht_his) / (2 * N)
    scalar_loss = torch.mean(L_prototype)
    return scalar_loss
