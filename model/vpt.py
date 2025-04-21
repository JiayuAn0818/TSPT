import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler

from .vision_transformer import VisionTransformer


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers == 0:
            self.mlp = nn.Identity()
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        # self.last_layer.weight_g.data.fill_(1)
        # if norm_last_layer:
        #     self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return  logits, x_proj

class NCD_ViT(VisionTransformer):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), Prompt_Token_num=1, VPT_type="Deep", 
                 basic_state_dict=None, nlayers=3, args=None):

        # Recreate ViT
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, norm_layer=norm_layer)

        self.nlayers = nlayers
        self.args = args
        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.VPT_type = VPT_type
        if VPT_type == "Deep":
            self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
        else:  # "Shallow"
            self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))

    def New_CLS_head(self, new_classes=15):
        self.head = DINOHead(in_dim=self.embed_dim, out_dim=new_classes, nlayers=self.nlayers)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.Prompt_Tokens.requires_grad = True
        try:
            for param in self.head.parameters():
                param.requires_grad = True
        except:
            pass

    def UnFreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'Prompt_Tokens': self.Prompt_Tokens}
        # print(prompt_state_dict)
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict):
        try:
            self.head.load_state_dict(prompt_state_dict['head'], False)
        except:
            print('head not match, so skip head')
        else:
            print('prompt head match')

        if self.Prompt_Tokens.shape == prompt_state_dict['Prompt_Tokens'].shape:

            # device check
            Prompt_Tokens = nn.Parameter(prompt_state_dict['Prompt_Tokens'].cpu())
            Prompt_Tokens.to(torch.device(self.Prompt_Tokens.device))

            self.Prompt_Tokens = Prompt_Tokens

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.Prompt_Tokens.shape)
            print('shape of model given prompt', prompt_state_dict['Prompt_Tokens'].shape)
            print('')

    def get_info(self, x):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        fea = self.norm(x)
        return fea

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":

            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            for i in range(len(self.blocks)):
                if self.args is not None:
                    if (i+1) > self.args.prompt_depth:
                        x = self.blocks[i](x)
                        continue
                # concatenate Prompt_Tokens
                Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                # firstly concatenate
                x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                num_tokens = x.shape[1]
                # lastly remove, a genius trick
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        else:  # self.VPT_type == "Shallow"
            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            # concatenate Prompt_Tokens
            Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            num_tokens = x.shape[1]
            # Sequntially procees
            x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x[:, 0]
        # use cls token for cls head
        features = torch.nn.functional.normalize(x, dim=1)
        logits, proj = self.head(features)
        return logits, proj, features
    def normalize_prototypes(self):
        w = self.head.last_layer.weight.data.clone()
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        self.head.last_layer.weight.data = w.clone()


class CIL_ViT(VisionTransformer):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), VPT_type="Deep", basic_state_dict=None,
                 layer_num=12, prompt_length=None, pool_size=None, top_k=1, head_type='token', use_prompt_mask=True,args=None):

        # Recreate ViT
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, norm_layer=norm_layer)
        
        self.use_prompt_mask = use_prompt_mask
        self.top_k = top_k
        self.prompt_length = prompt_length
        self.pool_size = pool_size
        self.depth = depth
        self.head_type = head_type
        self.layer_num = layer_num
        self.args = args

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        ## prompt initialization
        self.VPT_type = VPT_type
        if VPT_type == "Deep":
            self.Prompt_Keys = nn.Parameter(torch.zeros(pool_size, embed_dim))
            self.Prompt_Tokens = nn.Parameter(torch.zeros(layer_num, pool_size, prompt_length, embed_dim))
        else:  # "Shallow"
            self.Prompt_Keys = nn.Parameter(torch.zeros(pool_size, embed_dim))
            self.Prompt_Tokens = nn.Parameter(torch.zeros(1, pool_size, prompt_length, embed_dim))
        nn.init.uniform_(self.Prompt_Keys, -1, 1)  ## 初始化
        nn.init.uniform_(self.Prompt_Tokens, -1, 1)

    def expend_pos_embed(self, num_prefix_tokens=1):
        gs_new = self.patch_embed.grid_size
        posemb = self.pos_embed.data

        new_embed_len = self.patch_embed.num_patches + self.num_prefix_tokens + self.top_k * self.prompt_length
        posemb_new = torch.randn(1, new_embed_len, self.embed_dim)

        ntok_new = posemb_new.shape[1]
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]

        gs_old = int(math.sqrt(len(posemb_grid)))
        if ntok_new > gs_old ** 2:
            ntok_new -= gs_old ** 2
            # expand cls's pos embedding for prompt tokens
            posemb_prefix = posemb_prefix.expand(-1, ntok_new, -1)
        if not len(gs_new):  # backwards compatibility
            gs_new = [int(math.sqrt(ntok_new))] * 2
        assert len(gs_new) >= 2
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
        posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
        self.pos_embed = torch.nn.Parameter(posemb, requires_grad=True)

    def New_CLS_head(self, total_classes=15):
        self.head = nn.Linear(self.embed_dim, total_classes, bias=False)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.Prompt_Keys.requires_grad = True
        self.Prompt_Tokens.requires_grad = True
        try:
            for param in self.head.parameters():
                param.requires_grad = True
        except:
            pass

    def UnFreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'Prompt_Tokens': self.Prompt_Tokens}
        # print(prompt_state_dict)
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict):
        try:
            self.head.load_state_dict(prompt_state_dict['head'], False)
        except:
            print('head not match, so skip head')
        else:
            print('prompt head match')

        if self.Prompt_Tokens.shape == prompt_state_dict['Prompt_Tokens'].shape:

            # device check
            Prompt_Tokens = nn.Parameter(prompt_state_dict['Prompt_Tokens'].cpu())
            Prompt_Tokens.to(torch.device(self.Prompt_Tokens.device))

            self.Prompt_Tokens = Prompt_Tokens

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.Prompt_Tokens.shape)
            print('shape of model given prompt', prompt_state_dict['Prompt_Tokens'].shape)
            print('')

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def select_prompt(self, prompt_mask, cls_features=None):
        x_embed_mean = cls_features
        prompt_norm = self.l2_normalize(self.Prompt_Keys, dim=1) # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

        similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size

        if prompt_mask is None:
            _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
            ## batchwise_prompt
            prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            # print(prompt_id, id_counts)
            # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
            # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
            # Unless dimension is specified, this will be flattend if it is not already 1D.
            if prompt_id.shape[0] < self.pool_size:
                prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
            major_prompt_id = prompt_id[major_idx] # top_k
        else:
            major_prompt_id = prompt_mask # top_k

        # Put pull_constraint loss calculation inside
        batched_key_norm = prompt_norm[major_prompt_id.expand(cls_features.shape[0], -1)] # B, top_k, C
        x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * x_embed_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / cls_features.shape[0] # Scalar

        return major_prompt_id, reduce_sim

    def forward_features(self, x, cls_features, task_id=-1, train=False):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.use_prompt_mask and train:
            start = (task_id +1) * self.top_k
            end = (task_id + 2) * self.top_k
            single_prompt_mask = torch.arange(start, end).to(x.device)
            prompt_mask = single_prompt_mask
            if end > self.pool_size:
                prompt_mask = None
        else:
            prompt_mask = None
        

        idx, reduce_sim = self.select_prompt(prompt_mask, cls_features)

        if self.VPT_type == "Deep":
            cur_Prompt_Tokens = self.Prompt_Tokens[:, idx].reshape(self.layer_num, -1, self.embed_dim) # deep * (top_k * length) * C

            Prompt_Token_num = cur_Prompt_Tokens.shape[1]
            self.total_prompt_len = Prompt_Token_num

            for i in range(len(self.blocks)):
                if self.args is not None:
                    if (i+1) > self.args.prompt_depth:
                        x = self.blocks[i](x)
                        continue
                Prompt_Tokens = cur_Prompt_Tokens[i].unsqueeze(0)
                # firstly concatenate
                x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                num_tokens = x.shape[1]
                # lastly remove, a genius trick
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        else:  # self.VPT_type == "Shallow"
            cur_Prompt_Tokens = self.Prompt_Tokens[:, idx].reshape(1, -1, self.embed_dim)

            Prompt_Token_num = cur_Prompt_Tokens.shape[1]
            self.total_prompt_len = Prompt_Token_num
            # concatenate Prompt_Tokens
            Prompt_Tokens = cur_Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((Prompt_Tokens, x), dim=1)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)

            # concatenate CLS token
            x = torch.cat((cls_token, x), dim=1)
            x = self.pos_drop(x + self.pos_embed)
            x = self.blocks(x)
            # num_tokens = x.shape[1]
            # Sequntially procees
            # x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

        x = self.norm(x)
        return x, reduce_sim

    def forward_test(self, x, cls_features):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x_embed_mean = cls_features
        prompt_norm = self.l2_normalize(self.Prompt_Keys, dim=1) # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

        similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
        _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
        # print(torch.unique(idx,return_counts=True))

        if self.VPT_type == "Deep":
            cur_Prompt_Tokens = self.Prompt_Tokens[:, idx].reshape(self.layer_num, x.shape[0], -1, self.embed_dim)  # deep * (top_k * length) * C
            cur_Prompt_Tokens = cur_Prompt_Tokens.permute(1, 0, 2, 3)

            Prompt_Token_num = cur_Prompt_Tokens.shape[2]
            self.total_prompt_len = Prompt_Token_num

            for i in range(len(self.blocks)):
                Prompt_Tokens = cur_Prompt_Tokens[:, i]
                # firstly concatenate
                x = torch.cat((x, Prompt_Tokens), dim=1)
                num_tokens = x.shape[1]
                # lastly remove, a genius trick
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        x = self.norm(x)
        return x

    def forward(self, x, cls_features, task_id=-1, train=False):
        x, reduce_sim = self.forward_features(x, cls_features, task_id, train)

        if self.head_type == 'token':
            # use cls token for cls head
            x = x[:, 0, :] # fixme for old timm: x = self.pre_logits(x[:, 0, :])
        else:  # self.head_type == 'token'
            x = x[:, 1:(1 + self.total_prompt_len)]
            x = x.mean(dim=1)
            x = self.fc_norm(x)

        features = torch.nn.functional.normalize(x, dim=-1)
        logits = self.head(features)
        return logits, features, reduce_sim

    @torch.no_grad()
    def normalize_prototypes(self, task_id, args):
        w = self.head.weight.data.clone()
        if task_id==-1:
            w[-args.init_classes:] = torch.nn.functional.normalize(w[-args.init_classes:], dim=1, p=2)
        else:
            w[-args.per_stage_classes:] = torch.nn.functional.normalize(w[-args.per_stage_classes:], dim=1, p=2)
        # w = torch.nn.functional.normalize(w, dim=1, p=2)
        self.head.weight.data = w.clone()

