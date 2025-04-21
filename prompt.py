import math
import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler, Adam

from utils import get_params_groups, test
from models.loss import DistillLoss, info_nce_logits, SupConLoss
from models.vpt import NCD_ViT, CIL_ViT
from utils import cluster_acc


class CILPrompt:
    def __init__(self) -> None:
        pass
    
    def test(self, model, origin_model, test_loader, task_id, args):
        model.eval()
        origin_model.eval()
        preds, targets = [], []
        for batch_idx, (images, label, _) in enumerate(test_loader):
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                cls_features = origin_model.get_info(images)
                cls_features = cls_features[:, 0]
                output = model(images, cls_features, task_id)
                logits = output[0]
                preds.append(logits.argmax(1).cpu().numpy())
                targets.append(label.cpu().numpy())
        preds = np.concatenate(preds).reshape(-1, 1)
        targets = np.concatenate(targets).reshape(-1, 1)
        acc = cluster_acc(targets, preds)
        return acc

    def train_init(self, model_init, model_NCD, train_loader, test_loader, args):
        model_NCD.eval()
        model_init.Prompt_Tokens.data[:, :args.topk, :, :] = model_NCD.Prompt_Tokens.data.unsqueeze(1).clone()
        model_init.head.weight.data = model_NCD.head.last_layer.weight.data.clone()
        model_init.normalize_prototypes(-1, args)

        params_groups = get_params_groups(model_init)
        optimizer = Adam(params_groups, lr=args.lr)
        fp16_scaler = None
        if args.fp16:
            fp16_scaler = torch.cuda.amp.GradScaler()

        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.lr * 1e-3,
            )

        for epoch in range(args.epochs):
            model_init.train()
            for batch_idx, batch in enumerate(train_loader):
                (images, _), class_labels, uq_idxs = batch
                class_labels = class_labels.cuda(non_blocking=True)
                images = images.cuda(non_blocking=True)
                with torch.no_grad():
                    cls_features = model_NCD.get_info(images)
                    cls_features = cls_features[:, 0]
                with torch.cuda.amp.autocast(fp16_scaler is not None):
                    student_out, _, reduce_sim = model_init(images, cls_features, task_id=-1, train=True)

                    pull_loss = args.pull_constraint_coeff * reduce_sim

                    loss = 0
                    loss -= pull_loss
                    
                # Train acc
                optimizer.zero_grad()
                if fp16_scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    fp16_scaler.scale(loss).backward()
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                model_init.normalize_prototypes(-1, args)
                model_init.train()

            if epoch ==0 or (epoch+1)%10==0:
                acc = self.test(model_init, model_NCD, test_loader, -1, args)
                print('Train Epoch: {}, Init Classes ACC = {:.2f}.'.format(epoch+1, acc*100))
            # Step schedule
            exp_lr_scheduler.step()

        torch.save(model_init.state_dict(), args.OutputPathModels+'/CIL_init.npy')
    
    def exp_setting(self, args, task_id):
        model = CIL_ViT(prompt_length=args.prompt_num, VPT_type='Deep', pool_size=10, args=args)
        if task_id==0:
            model.New_CLS_head(args.init_classes)
            state_dict = torch.load(args.OutputPathModels+'/CIL_init.npy', map_location='cpu')
        else:
            model.New_CLS_head(args.init_classes + task_id * args.per_stage_classes)
            state_dict = torch.load(args.OutputPathModels+'/CIL_stage{}.npy'.format(task_id))
        model.load_state_dict(state_dict)
        fc_weight = model.head.weight.data.clone()
        model.head = nn.Linear(768, args.init_classes + (task_id+1) * args.per_stage_classes, bias=False)
        model.head.weight.data[:args.init_classes + task_id * args.per_stage_classes] = fc_weight
        model.Freeze()
        model.cuda()


        orign_model = NCD_ViT(Prompt_Token_num=args.prompt_num, VPT_type='Deep',)
        orign_model.New_CLS_head(args.per_stage_classes)
        if args.istest and task_id==args.stage:
            state_dict = torch.load(args.OutputPathModels+'/stage{}.npy'.format(task_id))
        else:
            state_dict = torch.load(args.OutputPathModels+'/stage{}.npy'.format(task_id+1))
        orign_model.load_state_dict(state_dict)
        orign_model.cuda()
        return model, orign_model

    def train_inc(self, input, train_loader, test_loader_all, task_id, args):
        model, model_NCD = input
        model.Prompt_Tokens.data[:, task_id+1:task_id+2, :, :] = model_NCD.Prompt_Tokens.data.unsqueeze(1).clone()
        model.head.weight.data[-args.per_stage_classes:] = model_NCD.head.last_layer.weight.data.clone()
        model.normalize_prototypes(task_id, args)

        params_groups = get_params_groups(model)
        optimizer = Adam(params_groups, lr=args.lr)
        fp16_scaler = None
        if args.fp16:
            fp16_scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.epochs):
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                (images, _), _, uq_idxs = batch
                images = images.cuda(non_blocking=True)
                with torch.no_grad():
                    cls_features = model_NCD.get_info(images)
                    cls_features = cls_features[:, 0]
                    pred_logits, _, _ = model_NCD(images)
                    _, pred_labels = pred_logits.max(1)
                    pred_labels += args.init_classes + task_id * args.per_stage_classes

                mask = range(args.init_classes + task_id * args.per_stage_classes, args.init_classes + (task_id+1) * args.per_stage_classes)
                not_mask = np.setdiff1d(np.arange(args.init_classes + (task_id+1) * args.per_stage_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).cuda()

                with torch.cuda.amp.autocast(fp16_scaler is not None):
                    student_out, _, reduce_sim = model(images, cls_features, task_id=task_id, train=True)
                    student_out = student_out.index_fill(dim=1, index=not_mask, value=float('-inf'))

                    cls_loss = nn.CrossEntropyLoss()(student_out, pred_labels)

                    pull_loss = args.pull_constraint_coeff * reduce_sim

                    loss = 0
                    loss += cls_loss 
                    loss -= pull_loss
                    
                # Train acc
                optimizer.zero_grad()
                if fp16_scaler is None:
                    loss.backward()
                    for n, p in model.head.named_parameters():
                        p.grad.data[:args.init_classes + task_id * args.per_stage_classes] *= 0
                    optimizer.step()
                else:
                    fp16_scaler.scale(loss).backward()
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                model.normalize_prototypes(task_id, args)
                model.train()

            if epoch ==0 or (epoch+1)%10==0:
                pstr = 'Train Epoch: {}, '.format(epoch+1)
                old = 0
                new = 0
                for i, test_loader in enumerate(test_loader_all):
                    acc = self.test(model, model_NCD, test_loader, i-1, args)
                    if i==0:
                        pstr += 'Init Classes ACC = {:.2f}, '.format(acc*100)
                        old = acc
                    else:
                        pstr += 'Stage{} Classes ACC = {:.2f}, '.format(i, acc*100)
                        new += acc
                print(pstr)
        torch.save(model.state_dict(), args.OutputPathModels+'/CIL_stage{}.npy'.format(task_id+1))
     