import argparse
import random
import os 
import numpy as np
import sys
from logger import Logger
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler

from models.vpt import NCD_ViT, CIL_ViT
from utils import test
from method.prompt import CILPrompt

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def exp_init(args):
    args.image_size = 224
    args.feat_dim = 768
    args.interpolation = 3
    args.crop_pct = 0.875
    args.n_views = 2

    if args.dataset == 'cifar100':
        from data.Cifar100 import get_cifar_100_datasets as get_dataset
    elif args.dataset == 'cub':
        from data.cub import get_cub_datasets as get_dataset
    elif args.dataset == 'tinyimagenet':
        from data.Tinyimagenet import get_tinyimagenet_datasets as get_dataset

    dataset = get_dataset(args)

    method = CILPrompt()
    return dataset, method


def main():
    print("START")
    args.OutputPathModels = os.path.join(OutputPath, 'model')
    if not os.path.exists(args.OutputPathModels):
        os.makedirs(args.OutputPathModels)
    dataset, method = exp_init(args)
    # --------------------
    # CIL Init Stage
    # --------------------
    train_dataset = dataset['init_train']
    test_dataset = dataset['init_test']
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=512, shuffle=True)

    model_init = CIL_ViT(prompt_length=args.prompt_num, VPT_type='Deep', pool_size=10, top_k=args.topk)
    state_dict = torch.load('./models/dino_vitbase16_pretrain.pth', map_location='cpu')
    model_init.load_state_dict(state_dict, False)
    model_init.New_CLS_head(args.init_classes)
    model_init.Freeze()
    model_init.cuda()

    model_NCD = NCD_ViT(Prompt_Token_num=args.prompt_num, VPT_type='Deep',)
    model_NCD.New_CLS_head(args.init_classes)
    model_NCD.load_state_dict(torch.load(args.OutputPathModels+'/init.npy'))
    for p in model_NCD.parameters():
        p.requires_grad = False
    model_NCD.cuda()
    
    model_init_path = args.OutputPathModels+'/CIL_init.npy'
    if os.path.exists(model_init_path) == False:
        method.train_init(model_init, model_NCD, train_loader, test_loader, args)
    # --------------------
    # CIL Incremental Stage
    # --------------------
    test_loader_all = []
    test_loader_all.append(test_loader)
    for task_id in range(args.stage):
        print('Stage {} Start!'.format(task_id+1))
        train_dataset = dataset['stage{}_train'.format(task_id+1)]
        test_dataset = dataset['stage{}_test'.format(task_id+1)]
        train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=256, shuffle=False)
        test_loader_all.append(test_loader)

        model_path = args.OutputPathModels+'/CIL_stage{}.npy'.format(task_id+1)
        if os.path.exists(model_path) == False:
            output = method.exp_setting(args, task_id)
            method.train_inc(output, train_loader, test_loader_all, task_id, args)

def summary_results():
    print("START TESTING")
    args.OutputPathModels = os.path.join(OutputPath, 'model')
    if not os.path.exists(args.OutputPathModels):
        os.makedirs(args.OutputPathModels)
    dataset, method = exp_init(args)
    ACC_Matrix = torch.zeros((args.stage+1, args.stage+1))
    test_loader_all = []
    for i in range(args.stage+1):
        if i==0:
            test_dataset = dataset['init_test']
            test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=512, shuffle=True, drop_last=False)
        else:
            test_dataset = dataset['stage{}_test'.format(i)]
            test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=512, shuffle=False, drop_last=False)
        test_loader_all.append(test_loader)

        model, orign_model = method.exp_setting(args, i)
        for j in range(i+1):
            acc = method.test(model, orign_model, test_loader_all[j], j-1, args)
            ACC_Matrix[i, j] = acc*100
    
    classes_num = [args.init_classes, args.per_stage_classes, args.per_stage_classes, args.per_stage_classes, args.per_stage_classes, args.per_stage_classes]
    for stage in range(args.stage+1):
        pstr = 'Stage: {}, '.format(stage)
        if stage==0:
            All = ACC_Matrix[0, 0]
            pstr += 'All = {:.2f}.'.format(All)
            print(pstr)
            continue
        New = ACC_Matrix[stage, stage]
        Old = 0
        for k in range(stage):
            Old += ACC_Matrix[stage, k] * classes_num[k]
        All = (Old + New*classes_num[stage])/sum(classes_num[:stage+1])
        Old /= sum(classes_num[:stage])
        pstr += 'Old = {:.2f} | New = {:.2f} | All = {:.2f} | '.format(Old, New, All)
        if stage == args.stage:
            BWT = 0
            Discovery = 0
            for k in range(stage):
                BWT += (ACC_Matrix[stage, k] - ACC_Matrix[k,k])*classes_num[k]
                if k>0:
                    Discovery += ACC_Matrix[stage, k]
            BWT /= sum(classes_num[:stage])
            Discovery = (Discovery+New)/args.stage
            pstr += 'BWT = {:.2f} | Discovery = {:.2f}.'.format(BWT, Discovery)
        print(pstr)

        

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='cncd', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--gpu', default='5', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--istest', action='store_true', default=False)

    parser.add_argument('--dataset', type=str, default='cifar100', help='options: cifar100')
    parser.add_argument('--log_name', default='frost', type=str)
    parser.add_argument('--stage', default=5, type=int)
    parser.add_argument('--init_classes', default=50, type=int)

    parser.add_argument('--grad_from_block', default=11, type=int)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=50, type=int)

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--sup_weight', type=float, default=0.35)

    parser.add_argument('--prompt_num', default=10, type=int)
    parser.add_argument('--topk', default=1, type=int)
    parser.add_argument('--pull_constraint_coeff', default=0.1, type=float)
    parser.add_argument('--prompt_depth', default=12, type=int)

    args = parser.parse_args()
    device = torch.device('cuda:'+args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # fix the seed for reproducibility
    seed = args.seed
    seed_torch(seed)

    OutputPath = os.path.join('./results_ours/{}'.format(args.dataset), args.log_name)
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)
    sys.stdout = Logger(output_dir=OutputPath, stream=sys.stdout)  # record log

    if args.istest:
        summary_results()
    else:
        main()
