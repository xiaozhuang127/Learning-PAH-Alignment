from __future__ import print_function
import torch.nn.functional as F
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, Predictor
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.myself_return_dataset import return_dataset
from myself_losses import BCE_softlabels, sigmoid_rampup, get_losses_unlabeled, get_loss, PrototypeLoss
from dwt2_tensor_mixup import source_mixup, mix_t_initial, mix_t_append, mix_feature_initial, mix_feature_append, \
    proto_count
from aa import AA

parser = argparse.ArgumentParser(description='UDA Classification')
parser.add_argument('--method', type=str, default='CDAC', choices=['CDAC'], help='CDAC is proposed method')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations to train (default: 50000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--lr_f', type=float, default=1.0, metavar='LR_F', help='learning rate (default: 1.0)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication(default: 0.1)')
parser.add_argument('--T', type=float, default=0.05, metavar='T', help='temperature')
parser.add_argument('--save_check', action='store_true', default=True, help='save checkpoint or not')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34', help='which network to use')
parser.add_argument('--source', type=str, default='synth', help='source domain')
parser.add_argument('--target', type=str, default='real', help='target domain')
parser.add_argument('--dataset', type=str, default='sample', help='the name of dataset')
parser.add_argument('--rampup_length', type=int, default=20000, help='ramp consistency loss weight up during first n training steps')
parser.add_argument('--rampup_coef', type=float, default=30.0, help='coefficient of consistency loss')
parser.add_argument('--topk', default=5, type=int, help='top-k indices of rank ordered feature elements')
parser.add_argument('--threshold', default=0.95, type=float, help='threshold of pseudo labeling')
parser.add_argument('--remark', type=str, default='', help='remark')
parser.add_argument('--lambda_u', default=1, type=float, help='coefficient of AAC')
parser.add_argument('--th', default=0.5, type=float, help='High frequency mixing coefficients based on wavelet transform')
parser.add_argument('--ways', type=int, default=10, help='number of classes sampled')
parser.add_argument('--src_shots', type=int, default=24, help='number of samples per source classes')
parser.add_argument('--trg_shots', type=int, default=1, help='number of samples per target classes')
parser.add_argument('--alpha', type=float, default=0.01, help='loss weight')
parser.add_argument('--beta', type=float, default=10, help='loss weight')
parser.add_argument('--grc', type=float, default=1, help='loss weight')
parser.add_argument('--group_consistency', default=True, help='use group_consistency loss with given weight')
parser.add_argument('--high_mix', default=True, help='high_mix')
parser.add_argument('--mix_add', default=True, help='mix_add')
parser.add_argument('--proto_use', default=True, help='proto_use')
parser.add_argument('--proto_add', default=True, help='proto_add')
parser.add_argument('--tau', default=2, type=float, help='threshold of constrative loss')

args = parser.parse_args()

torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('Dataset %s Source %s Target %s reliable pseudo-labels perclass %s Network %s' % (
    args.dataset, args.source, args.target, num, args.net))
source_dataset, target_dataset, target_dataset_unl, target_dataset_test, class_list = return_dataset(
    args)


def compute_archetypes(features, labels, device):
    num_classes = 10
    archetypes = []
    for label in range(num_classes):
        class_features = features[labels == label].to(device)
        if class_features.size(0) > 0:
            model = AA(n_archetypes=1, device=device, max_iter=300, tol=1e-4, init="uniform",
                         init_kwargs=None, save_init=False, verbose=False,
                         method="autogd", method_kwargs={"optimizer": "SGD", "optimizer_kwargs": {"lr": 1e-3}})
            model.fit_transform(class_features)
            archetype = torch.tensor(model.archetypes_, device=device).squeeze()
        else:
            archetype = torch.zeros(512, device=device)
        archetypes.append(archetype)
    archetypes = torch.stack(archetypes, dim=0)
    return archetypes


def save_pytorch_encoder_weights(G, save_path):
    weights = {}
    for name, param in G.named_parameters():
        if param.requires_grad:
            weights[name] = param.data.cpu().numpy()
    np.savez(save_path, **weights)


def folder_preparation(args):
    import datetime
    nowtime = datetime.datetime.now().strftime('%m%d%H%M%S')
    main_path = 'record/sample/alpha=%.2f_th=%.1f_grc=%.1f_x_%s_%s_%s_net_%s_%s_to_%s_num_%s_%s' % (
            args.alpha, args.th, args.grc, nowtime, args.dataset, args.method, args.net, args.source,
            args.target, num, args.remark)
    logs_file = os.path.join(main_path, 'logs')
    loss_file = os.path.join(main_path, 'loss')
    checkpath = os.path.join(main_path, 'checkpath')
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    if not os.path.exists(checkpath):
        os.makedirs(checkpath)
    return main_path, logs_file, loss_file, checkpath

main_path, logs_file, loss_file, checkpath = folder_preparation(args)
print("Main path to save: {}".format(main_path))

if args.net == 'resnet34':
    G = resnet34(pretrained=False)
    inc = 512
    bs = 16
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
    bs = 32
else:
    raise ValueError('Model cannot be recognized.')

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)
weights_init(F1)

G = nn.DataParallel(G)
F1 = nn.DataParallel(F1)
G = G.to(device)
F1 = F1.to(device)

opt = {}
opt["logs_file"] = logs_file
opt["loss_file"] = loss_file
opt["checkpath"] = checkpath
opt["class_list"] = class_list

lr = args.lr
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs, num_workers=0, shuffle=True, drop_last=True)
target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=len(target_dataset), num_workers=0,
                                            shuffle=True, drop_last=True)
target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs * 2, num_workers=0, shuffle=True,
                                                drop_last=True)
target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs * 2, num_workers=0, shuffle=True,
                                                 drop_last=True)
save_pytorch_encoder_weights(G, "pytorch_encoder_weights.npz")


def train(device, opt):
    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=args.lr_f, momentum=0.9, weight_decay=0.0005, nesterov=True)
    torch.autograd.set_detect_anomaly(True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)

    best_acc = 0

    BCE = BCE_softlabels().to(device)  
    criterion = nn.CrossEntropyLoss().to(device)  

    start_time = time.time()
    for step in range(all_step):
        rampup = sigmoid_rampup(step, args.rampup_length)
        w_cons = args.rampup_coef * rampup

        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr_f = optimizer_f.param_groups[0]['lr']
        lr_g = optimizer_g.param_groups[0]['lr']

        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)
        
        x_t, target_t, x_bar_t = data_t[0], data_t[1], data_t[3]  
        im_data_t = x_t.to(device)
        gt_labels_t = target_t.to(device)
        im_data_bar_t = x_bar_t.to(device)
       
        x_s, target_s, x_bar_s = data_s[0], data_s[1], data_s[3]  
        if step % len_train_target_semi == 0 and args.high_mix:
            mix_t = mix_t_initial(x_t, target_t, False)  
        if args.high_mix:
            x_s = source_mixup(x_s, target_s, mix_t, args.th) 
        im_data_s = x_s.to(device)
        gt_labels_s = target_s.to(device)
        im_data_bar_s = x_bar_s.to(device)

        x_tu, target_u, x_bar_tu, x_bar2_tu = data_t_unl[0], data_t_unl[1], data_t_unl[3], data_t_unl[4]
        im_data_tu = x_tu.to(device)
        im_data_bar2_tu = x_bar2_tu.to(device)
        target_u = target_u.to(device)

        zero_grad_all()
        data = torch.cat((im_data_s, im_data_t), 0)
        target = torch.cat((gt_labels_s, gt_labels_t), 0)
        output = G(data)
        out1 = F1(output)
        ce_loss = criterion(out1, target.long())
        if step % len_train_target_semi == 0 and args.proto_use:
            data_t_feature = output[bs:].detach()
            mix_feature_tensor, labels_t = mix_feature_initial(data_t_feature, gt_labels_t, False, device)
            labels_tensor = torch.tensor(labels_t, dtype=torch.long)

        mix_feature_tensor = mix_feature_tensor.detach()
        labels_tensor = labels_tensor.to(device)
        mix_feature_tensor = mix_feature_tensor.to(device)
        data_s_feature = None
        if (args.alpha != 0 or args.beta != 0) and step % len_train_target_semi == 0 and args.proto_use:
            proto_t = proto_count(mix_feature_tensor, labels_tensor, 10, 512)
        proto_loss = 0
        if args.alpha != 0:
            data_s_feature = output[:bs].detach()
            data_s_feature = data_s_feature.to(device)
            gt_labels_s = gt_labels_s.to(device)
            proto_s = proto_count(data_s_feature, gt_labels_s, 10, 512)
            criterion_aux = PrototypeLoss(ways=args.ways, trg_shots=args.trg_shots, src_shots=args.src_shots)
            proto_loss = criterion_aux(data_s_feature, proto_s.detach(), gt_labels_s.long())

        archetypes_s = compute_archetypes(data_s_feature, gt_labels_s, device=device)
        archetypes_t = compute_archetypes(mix_feature_tensor, labels_tensor, device=device)
        arche_loss = get_loss(proto_t, archetypes_t, proto_s, archetypes_s, args.tau)

        first_loss = 0
        first_loss = ce_loss + args.alpha * arche_loss + args.beta * proto_loss

        first_loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()
        pl_loss, group_loss, un_img_list, un_img_feature, un_target_list = get_losses_unlabeled(args, G, F1,
                                                                                                im_data=im_data_tu,
                                                                                                im_data_bar2=im_data_bar2_tu,
                                                                                                targets_u=target_u,
                                                                                                im_data_t=im_data_t,
                                                                                                gt_labels_t=gt_labels_t,
                                                                                                group_consistency=args.group_consistency,
                                                                                                step=step,
                                                                                                enhanced=False,
                                                                                                mix_add=args.mix_add,
                                                                                                proto_add=args.proto_add)
        un_img_list = un_img_list.numpy()
        un_img_feature = un_img_feature.numpy()
        mix_t = mix_t_append(mix_t, un_img_list, un_target_list) 
        mix_feature_tensor, labels_tensor = mix_feature_append(mix_feature_tensor, labels_tensor, un_img_feature,
                                                               un_target_list, device)
        loss = pl_loss + args.grc * group_loss
        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()

        loss_train = '{:.6f} {:.6f}\n'.format(ce_loss, pl_loss)
        with open(opt["loss_file"], 'a') as f:
            f.write(loss_train)

        if step % args.log_interval == 0:
            log_train = 'S {} T {} Train Ep: {} lr_f{:.6f} lr_g{:.6f}\n'.format(args.source, args.target,
                                                                                step, lr_f, lr_g)
            with open(opt["logs_file"], 'a') as f:
                f.write(log_train)

        if (step % args.save_interval) == 0 and step > 0 or (step == all_step - 1):
            loss_test, acc_test = test(target_loader_test)
            G.train()
            F1.train()
            if acc_test >= best_acc:
                best_acc = acc_test

            print('Current acc test %f best acc test %f ' % (
                acc_test, best_acc))

            with open(opt["logs_file"], 'a') as f:
                f.write('step %d current %f best %f.\n' % (
                    step, acc_test, best_acc))
            G.train()
            F1.train()
train(device=device, opt=opt)
