import math
from math import lgamma
import numpy as np
from sklearn import manifold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy

import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler

from utils import get_params_groups, test
from models.loss import DistillLoss, info_nce_logits, SupConLoss
from models.vpt import NCD_ViT, CIL_ViT

# 计算簇的特征中心
def compute_class_centers(features, labels, args):
    n_classes = args.per_stage_classes
    class_centers = np.zeros((n_classes, features.shape[1]))
    for label in range(n_classes):
        class_samples = features[labels == label]
        center = np.mean(class_samples, axis=0)
        norm = np.linalg.norm(center)
        center = center/norm
        class_centers[label] = center
    return class_centers

class Ours:
    def __init__(self) -> None:
        pass

    def visual_save_feature(self, model, train_loader, args):
        model.eval()
        all_feat = []
        all_raw_feat = []
        all_proj = []
        all_labels = []
        for batch_idx, (images, label, _) in enumerate(train_loader):
            images = images[0].cuda(non_blocking=True)
            with torch.no_grad():
                _, proj, feat = model(images)
                raw_feat = model.get_info(images)
                all_feat.append(feat.detach().clone().cuda())
                all_raw_feat.append(raw_feat[:, 0].detach().clone().cuda())
                all_proj.append(proj.detach().clone().cuda())
                all_labels.append(label.detach().clone().cuda())
        all_feat = torch.cat(all_feat, dim=0).cpu().numpy()
        all_raw_feat = torch.cat(all_raw_feat, dim=0).cpu().numpy()
        all_proj = torch.cat(all_proj, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        np.save( args.OutputPathModels+'/stage1_init_feature.npy', all_feat)
        np.save( args.OutputPathModels+'/stage1_init_raw_feature.npy', all_raw_feat)
        np.save(args.OutputPathModels+'/stage1_init_label.npy', all_labels)
        np.save(args.OutputPathModels+'/stage1_init_proj.npy', all_proj)

    def visual_tsne(self, args):
        MOUSE_10X_COLORS = ["#9e80d4","#87ceeb","#ff8033","#ffc0cb","#36bf36","#ff6347","#f0e68c","#800080","#00bfff","#ff1493",]
         
        fea = np.load(args.OutputPathModels+'/stage1_init_feature.npy')
        raw_fea = np.load(args.OutputPathModels+'/stage1_init_raw_feature.npy')
        label = np.load(args.OutputPathModels+'/stage1_init_label.npy')
        proj = np.load(args.OutputPathModels+'/stage1_init_proj.npy')
        ################tsne画图nooverlap_eegnet###########################
        tsne = manifold.TSNE(n_components=2,perplexity=135,learning_rate=100, n_iter=300, random_state=101)
        # ef_no_X_1 = tsne.fit_transform(fea)
        ef_no_X_1 = tsne.fit_transform(raw_fea)

        #=================画图==============
        plt.figure(figsize=(10,10))
        class_labels = np.unique(label)
        for idx, i in enumerate(class_labels):
            index = (label==i)
            plt.scatter(ef_no_X_1[index,0], ef_no_X_1[index,1],c=MOUSE_10X_COLORS[idx], label=i)

        plt.yticks(size=20)
        plt.xticks(size=20)
        # plt.savefig('./tsne_feature', dpi=300)
        plt.savefig('./tsne_raw_features', dpi=300)

    def visual_kmeans(self, args):
        pca = PCA(n_components=2)
        features  = np.load(args.OutputPathModels+'/stage1_init_feature.npy')
        raw_fea = np.load(args.OutputPathModels+'/stage1_init_raw_feature.npy')
        labels = np.load(args.OutputPathModels+'/stage1_init_label.npy')
        proj = np.load(args.OutputPathModels+'/stage1_init_proj.npy')

        # 使用KMeans进行聚类
        kmeans = KMeans(n_clusters=10, random_state=42)
        kmeans.fit(raw_fea)
        
        # 获取聚类结果
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        tsne = manifold.TSNE(n_components=2,perplexity=135,learning_rate=100, n_iter=300, random_state=101)
        ef_no_X_1 = tsne.fit_transform(raw_fea)

        # 可视化结果
        plt.figure(figsize=(10,10))
        # 绘制每个数据点
        plt.scatter(ef_no_X_1[:, 0], ef_no_X_1[:, 1], c=labels, cmap='tab10', alpha=0.6)
        
        plt.savefig('./tsne_kmeans_raw_features', dpi=300)
        # 计算每个簇的数量
        cluster_sizes = np.bincount(labels)

        # 打印每个簇的数量
        for i, size in enumerate(cluster_sizes):
            print(f"Cluster {i}: {size} points")

    def exp_setting(self, args, task_id):
        model = NCD_ViT(Prompt_Token_num=args.prompt_num, VPT_type='Deep',)
        # if task_id==0:
        model.New_CLS_head(args.init_classes)
        state_dict = torch.load(args.OutputPathModels+'/init.npy', map_location='cpu')
        # else:
        #     model.New_CLS_head(args.per_stage_classes)
        #     state_dict = torch.load(args.OutputPathModels+'/stage{}.npy'.format(task_id))
        model.load_state_dict(state_dict)
        # model.head.last_layer = nn.utils.weight_norm(nn.Linear(768, args.per_stage_classes, bias=False))
        # model.head.last_layer.weight_g.data.fill_(1)
        model.head.last_layer = nn.Linear(768, args.per_stage_classes, bias=False)
        model.Freeze()
        model.cuda()
        return model
    
    def train_init(self, model_init, train_loader, test_loader, args):
        params_groups = get_params_groups(model_init)
        optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        fp16_scaler = None
        if args.fp16:
            fp16_scaler = torch.cuda.amp.GradScaler()

        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.lr * 1e-3,
            )


        cluster_criterion = DistillLoss(
                            args.warmup_teacher_temp_epochs,
                            args.epochs,
                            args.n_views,
                            args.warmup_teacher_temp,
                            args.teacher_temp,
                        )
        model_init.normalize_prototypes()

        for epoch in range(args.epochs):
            model_init.train()
            for batch_idx, batch in enumerate(train_loader):
                images, class_labels, uq_idxs = batch
                class_labels = class_labels.cuda(non_blocking=True)
                images = torch.cat(images, dim=0).cuda(non_blocking=True)

                with torch.cuda.amp.autocast(fp16_scaler is not None):
                    student_out, student_proj, _ = model_init(images)
                    teacher_out = student_out.detach()

                    # clustering, sup
                    sup_logits = torch.cat([f for f in (student_out / 0.1).chunk(2)], dim=0)
                    sup_labels = torch.cat([class_labels for _ in range(2)], dim=0)
                    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                    # clustering, unsup
                    cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                    me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                    cluster_loss += args.memax_weight * me_max_loss

                    # represent learning, unsup
                    contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                    contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                    # representation learning, sup
                    student_proj = torch.cat([f.unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                    student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                    sup_con_labels = class_labels
                    sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                    # pstr = ''
                    # pstr += f'cls_loss: {cls_loss.item():.4f} '
                    # pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                    # pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                    # pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                    loss = 0
                    loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                    loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                    
                # Train acc
                optimizer.zero_grad()
                if fp16_scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    fp16_scaler.scale(loss).backward()
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                model_init.normalize_prototypes()
                model_init.train()


            if epoch ==0 or (epoch+1)%10==0:
                acc = test(model_init, test_loader, args)
                print('Train Epoch: {}, Init Classes ACC = {:.2f}.'.format(epoch+1, acc*100))
            # Step schedule
            exp_lr_scheduler.step()

        torch.save(model_init.state_dict(), args.OutputPathModels+'/init.npy')

    def split_cluster(self, cluster_features, args):
        # 将当前簇分裂成两个簇
        kmeans_split = KMeans(n_clusters=2)
        kmeans_split.fit(cluster_features)
        # 获取分裂后的簇标签
        split_cluster_labels = kmeans_split.labels_
        split_cluster_centers = kmeans_split.cluster_centers_
        # 计算每个簇的数量
        split_cluster_sizes = np.bincount(split_cluster_labels)
        split_sorted_cluster_indices = np.argsort(split_cluster_sizes)[::-1]
        if split_cluster_sizes[split_sorted_cluster_indices[0]] > (args.samples_per_class):
            delete_num = int(split_cluster_sizes[split_sorted_cluster_indices[0]]-args.samples_per_class)
            # 获取当前簇的样本索引
            cluster_samples_idx = np.where(split_cluster_labels == split_sorted_cluster_indices[0])[0]
            cluster_samples = cluster_features[cluster_samples_idx]

            # 计算每个样本到簇中心的距离（使用欧式距离）
            distances = np.linalg.norm(cluster_samples - split_cluster_centers[split_sorted_cluster_indices[1]], axis=1)

            # 按距离排序，找到最远的 k 个点
            nearest_indices = np.argsort(distances)[:delete_num]  # 获取距离最大的 k 个索引
            # 剔除最远的 k 个点
            remaining_samples_idx = np.delete(cluster_samples_idx, nearest_indices)
            return cluster_features[remaining_samples_idx], split_cluster_labels[remaining_samples_idx]
        else:
            cluster_samples_idx = np.where(split_cluster_labels == split_sorted_cluster_indices[0])[0]
            return cluster_features[cluster_samples_idx], split_cluster_labels[cluster_samples_idx]
            
    def initialization_w_cluster(self, model, train_loader, test_loader, task_id, args):
        args.samples_per_class = 500 # cifar 100
        model.eval()
        all_feat = []
        all_labels = []
        for batch_idx, (images, label, _) in enumerate(train_loader):
            images = images[0].cuda(non_blocking=True)
            with torch.no_grad():
                _, proj, feat = model(images)
                all_feat.append(feat.detach().clone().cuda())
                all_labels.append(label.detach().clone().cuda())
        all_feat = torch.cat(all_feat, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

        # unique_labels, counts = np.unique(all_labels, return_counts=True)
        # # 输出不同类别的标签和它们的数量
        # for label, count in zip(unique_labels, counts):
        #     print(f"类别 {label}: {count} 个样本")

        # 初始化存储分裂簇的特征
        split_features = []
        split_labels = []
        cluster_num = args.per_stage_classes
        left_feature = all_feat.copy()
        while cluster_num > 1:
            kmeans = KMeans(n_clusters=cluster_num)
            kmeans.fit(left_feature)
            labels = kmeans.labels_
            center = kmeans.cluster_centers_
            # 计算每个簇的数量
            cluster_sizes = np.bincount(labels)
            # 对簇按样本数排序（从大到小）
            sorted_cluster_indices = np.argsort(cluster_sizes)[::-1]
            if cluster_sizes[sorted_cluster_indices[0]] > (args.samples_per_class*args.alpha):
                # 获取当前簇的样本
                cluster_samples_idx = np.where(labels == sorted_cluster_indices[0])[0]
                cluster_features = left_feature[cluster_samples_idx]
                # 将当前簇分裂成两个簇
                kmeans_split = KMeans(n_clusters=2)
                kmeans_split.fit(cluster_features)
                # 获取分裂后的簇标签
                split_cluster_labels = kmeans_split.labels_
                split_features.append(cluster_features)
                split_labels.append(split_cluster_labels+cluster_num-2)
                remain_features = cluster_features
                cluster_num -= 2
            else:
                split_features.append(left_feature)
                split_labels.append(labels)
                break
            mask = np.isin(left_feature, remain_features).all(axis=1)
            left_feature = left_feature[~mask]
            
        # split_features.append(remain_features)
        # split_labels.append(np.ones_like(remain_labels)*(cluster_num-1))
        split_features = np.concatenate(split_features, axis=0)
        split_labels = np.concatenate(split_labels, axis=0)

        updated_labels = np.zeros(len(all_feat), dtype=split_labels.dtype)
        for i, fea in enumerate(split_features):
            idx = np.where(np.all(all_feat == fea, axis=1))[0]
            if len(idx) > 0:
                updated_labels[idx[0]] = split_labels[i]  # 更新标签

        init_center = compute_class_centers(all_feat, updated_labels, args, task_id) 
        return torch.tensor(init_center).cuda()

    def train_inc(self, model, train_loader, test_loader, task_id, args):
        init_center = self.initialization_w_cluster(model, train_loader, test_loader, task_id, args)
        model.head.last_layer.weight.data = init_center.to(torch.float32)
        params_groups = get_params_groups(model)
        optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        fp16_scaler = None
        if args.fp16:
            fp16_scaler = torch.cuda.amp.GradScaler()

        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.lr * 1e-3,
            )


        cluster_criterion = DistillLoss(
                            args.warmup_teacher_temp_epochs,
                            args.epochs,
                            args.n_views,
                            args.warmup_teacher_temp,
                            args.teacher_temp,
                        )
        model.normalize_prototypes()

        num_sample = len(train_loader.dataset)
        fea_bank = torch.randn(num_sample, 768).cuda()
        score_bank = torch.randn(num_sample, args.per_stage_classes).cuda()
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                images, class_labels, uq_idxs = batch
                x = images[0].cuda()
                outputs, _, fea = model(x)
                outputs = nn.Softmax(-1)(outputs/0.1)
                fea_bank[uq_idxs] = fea.detach().clone()
                score_bank[uq_idxs] = outputs.detach().clone()  #.cpu()

        for epoch in range(args.epochs):
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                images, class_labels, uq_idxs = batch
                class_labels = class_labels.cuda(non_blocking=True)
                images = torch.cat(images, dim=0).cuda(non_blocking=True)

                with torch.cuda.amp.autocast(fp16_scaler is not None):
                    student_out, student_proj, student_fea = model(images)
                    teacher_out = student_out.detach()

                    # clustering, unsup
                    cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                    me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                    cluster_loss += args.memax_weight * me_max_loss

                    # represent learning, unsup
                    contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                    contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                    fea, fea_bar = student_fea.chunk(2)
                    with torch.no_grad():
                        distance = fea @ fea_bank.t()
                        dist_value, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.neighbor)
                        dist_value = dist_value[:, 1:]
                        idx_near = idx_near[:, 1:]
                        score_near = score_bank[idx_near]  
                        score_near = score_near.permute(0, 2, 1)
                        fea_bank[uq_idxs] = fea.detach().clone()
                        softmax_out = nn.Softmax(dim=1)(student_out.chunk(2)[0]/0.1)
                        output_re = softmax_out.unsqueeze(1)
                        score_bank[uq_idxs] = softmax_out.detach().clone()  #.cpu()
                    const = torch.bmm(torch.log(torch.bmm(output_re, score_near)), dist_value.unsqueeze(2)).squeeze()
                    loss_const = -torch.mean(const)

                    loss = 0
                    loss += cluster_loss
                    loss += contrastive_loss
                    loss += loss_const
                    
                # Train acc
                optimizer.zero_grad()
                if fp16_scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    fp16_scaler.scale(loss).backward()
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                model.normalize_prototypes()
                model.train()

            if epoch ==0 or (epoch+1)%10==0:
                acc = test(model, test_loader, args)
                print('Train Epoch: {}, Stage{} Classes ACC = {:.2f}.'.format(epoch+1, task_id+1, acc*100))
            # Step schedule
            exp_lr_scheduler.step()

        torch.save(model.state_dict(), args.OutputPathModels+'/stage{}.npy'.format(task_id+1))
