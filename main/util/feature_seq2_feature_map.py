import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from util.misc import NestedTensor, is_main_process


def loc2mask(loc: Tensor):
    bs, seq_len, _ = loc.shape
    ret_map = torch.zeros((bs,))
""" 
"""
def convert_feature_seq2_map(feature_seq: Tensor, loc: Tensor, seq_mask: Tensor, map_size: int, is_train=False):
    bs = loc.size(0)
    seq_len = loc.size(1)
    out = []
    for scale_name, scale_feature in sorted(feature_seq.items()):

        # scale_feature = torch.nn.functional.interpolate(scale_feature, scale_factor=0.5, recompute_scale_factor=False)

        scale_feature = torch.mean(scale_feature, dim=(2,3), keepdim=True)

        _, channel, h, w = scale_feature.shape
        scale_feature = scale_feature.view(-1, seq_len, channel, h, w)

        # ret_map = torch.zeros((bs, c, h * map_size, w * map_size))

        ret_map = torch.zeros((bs, channel, h * map_size, w * map_size)).to(scale_feature.device)
        padding_mask = torch.zeros((bs, h * map_size, w * map_size)).to(scale_feature.device)

        # global_feature_map = torch.mean(scale_feature, dim=1)

        b_list = []
        s_list = []
        r_list = []
        c_list = []
        filled = False
        for b_idx in range(bs):
            for seq_idx in range(seq_len):
                # if seq_mask[b_idx, seq_idx] == 0:
                #     continue
                r, c = loc[b_idx, seq_idx]
                r, c = int(r), int(c)
                if c >= map_size or r >= map_size:
                    continue

                b_list.append(b_idx)
                s_list.append(seq_idx)
                r_list.append(r)
                c_list.append(c)
                filled = True
                ret_map[b_idx, :, r*h:(r+1)*h, c*w:(c+1)*w] = ret_map[b_idx, :, r*h:(r+1)*h, c*w:(c+1)*w] + scale_feature[b_idx, seq_idx]
                # ret_map[b_idx, :, r * h:(r + 1) * h, c * w:(c + 1) * w] = scale_feature[b_idx, seq_idx]
                padding_mask[b_idx, r*h:(r+1)*h, c*w:(c+1)*w] = 1

        # ret_map[b_list, :, r_list, c_list] = scale_feature
        if not filled:
            print(loc[:30])

        # hack
        # ret_map[:, :, :h, :w] = global_feature_map
        # padding_mask[:, :h, :w] = 1
        padding_mask = padding_mask.float()
        ret_map.to(scale_feature.device)
        padding_mask.to(scale_feature.device)

        # if is_train:
        #     k = torch.randint(low=0, high=4, size=(1,)).item()
        #     padding_mask = torch.rot90(padding_mask, k, (1, 2))
        #     ret_map = torch.rot90(ret_map, k, (2, 3))

        padding_mask = padding_mask.bool()
        mask_b, mask_h, mask_w = padding_mask.shape
        # ret_map[~padding_mask[:, None,].expand(mask_b, channel, mask_h, mask_w)] = 0 * ret_map[~padding_mask[:, None,].expand(mask_b, channel, mask_h, mask_w)]
        # print(ret_map.shape)
        nt = NestedTensor(ret_map, padding_mask).to(scale_feature.device)
        out.append(nt)

    return out
