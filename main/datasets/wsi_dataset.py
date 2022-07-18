from pathlib import Path

import torch
import os.path as osp
import torch.utils.data as data_utils
from util.misc import get_local_rank, get_local_size, NestedTensor
import datasets.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import json
import cv2
import re
from typing import Tuple
import numpy as np



def parse_patch_fname(fp: str) -> Tuple[int, int]:
    # d15-08498-tile-r16896-c20480-512x512
    tmp = fp
    #fp = osp.basename(fp).rsplit('.', 1)[0]
    try:
        psize = int(fp.rsplit('x', 1)[-1])
    except:
        print(fp, tmp)
    r_loc = int(re.findall(r'(?<=r)\d+', fp)[0])
    c_loc = int(re.findall(r'(?<=c)\d+', fp)[0])

    r_loc = r_loc // psize
    c_loc = c_loc // psize
    return (r_loc, c_loc)

class WSIDataset(data_utils.Dataset):
    def __init__(self, ann_file=None, transforms=None, num_class=4,
                 cache_mode=False, local_rank=0, local_size=1
                 ):
        super().__init__()

        with open(ann_file, 'rt') as infile:
            data = json.load(infile)

        # (pid, label, [patch_fp])
        self.wsi_loc = []
        for k, v in data.items():
            pid = k
            label = v['label']
            fp = v['patch']
            self.wsi_loc.append((pid, label, fp))

        sorted(self.wsi_loc, key=lambda x: x[0])

        self.num_class = num_class

        self.pid_2_img_id = {}
        for idx, v in enumerate(self.wsi_loc):
            self.pid_2_img_id[v[0]] = idx

        self._transforms = transforms
        self.bag_len = 800

    def __len__(self):
        # return 80
        return len(self.wsi_loc)

    def __getitem__(self, index):
        image_id, label, patch_list = self.wsi_loc[index]
        if self.num_class == 2:
            label = torch.tensor(label > 0).long()
        else:
            label = torch.tensor(label)
        patch_ims = [] #[torch.zeros((3, 512, 512)), torch.zeros((3, 512, 512))]
        patch_loc = []
        for fp in patch_list:
            im = cv2.imread(fp)[:, :, ::-1]
            patch_ims.append(im)
            patch_loc.append(parse_patch_fname(fp))

        min_r = min([x[0] for x in patch_loc])
        min_c = min([x[1] for x in patch_loc])

        patch_loc = [(x[0] - min_r, x[1] - min_c) for x in patch_loc]

        # max_r = max([x[0] for x in patch_loc])
        # max_c = max([x[1] for x in patch_loc])

        if self._transforms is not None:
            patch_ims = [self._transforms(image=x)['image'] for x in patch_ims]

        # N * C * H * W
        patch_bag = torch.stack(patch_ims)

        patch_len = len(patch_bag)

        # random select
        if patch_len > self.bag_len:
            idx = np.random.choice(patch_len, size=self.bag_len, replace=False)
            patch_bag = patch_bag[idx]
            patch_loc = [patch_loc[ix] for ix in idx]


        # print(patch_bag.shape)

        n, c, h, w = patch_bag.shape
        pat_zeros = torch.zeros((self.bag_len - n, c, h, w))

        ret = torch.cat([patch_bag, pat_zeros])
        mask = torch.zeros((self.bag_len))
        # mask[:, :n] = 1
        mask[:n] = 1
        
        loc = torch.zeros((self.bag_len, 2))

        for l_idx in range(len(patch_loc)):
            loc[l_idx] = torch.tensor(patch_loc[l_idx])

        pid_img_id = self.pid_2_img_id[image_id]
        target = {
            'label': label,
            'pid': torch.tensor(pid_img_id)
        }

        return ret, loc, mask, target
        # return NestedTensor(ret, mask=mask), label


class WSIFeautreMapDataset(data_utils.Dataset):
    def __init__(self, img_folder=None, ann_file=None, transforms=None, cache_mode=False, local_rank=0, local_size=1):
        super().__init__()

        self._transforms = transforms
        self.bag_len = 30

    def __len__(self):
        return 300

    def __getitem__(self, index):
        # image_id = self.ids[index]
        label = torch.tensor(0).long()
        patch_ims = [torch.zeros((3, 512, 512)), torch.zeros((3, 512, 512))]

        # if self._transforms is not None:
        #     patch_ims = [self._transforms(x) for x in patch_ims]

        # N * C * H * W
        patch_bag = torch.stack(patch_ims)
        n, c, h, w = patch_bag.shape
        pat_zeros = torch.zeros((self.bag_len - n, c, h, w))

        ret = torch.cat([patch_bag, pat_zeros])
        mask = torch.zeros((self.bag_len))
        # mask[:, :n] = 1
        mask[:n] = 1

        loc = torch.zeros((self.bag_len, 2))

        # debug
        rows = 3
        for r_idx in range(rows):
            for c_idx in range(3):
                layer_idx = r_idx * rows + c_idx
                loc[layer_idx] = torch.tensor([r_idx, c_idx])

        return ret, loc, mask, label

def make_wsi_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':

        return A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Resize(256, 256),
            # A.ColorJitter(),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16,
                            min_holes=1, min_height=8, min_width=8),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.augmentations.transforms.Normalize(),
            A.pytorch.transforms.ToTensorV2(),
        ])

        # return T.Compose([
        #     T.RandomHorizontalFlip(),
        #     T.RandomSelect(
        #         T.RandomResize(scales, max_size=1333),
        #         T.Compose([
        #             T.RandomResize([400, 500, 600]),
        #             T.RandomSizeCrop(384, 600),
        #             T.RandomResize(scales, max_size=1333),
        #         ])
        #     ),
        #     normalize,
        # ])

    if image_set == 'val' or image_set == 'test':
        return A.Compose([
            A.Resize(256, 256),
            A.augmentations.transforms.Normalize(),
            A.pytorch.transforms.ToTensorV2(),
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):

    fold_split = args.FOLD_SPLIT
    seed = args.DATASET.DATASET_SEED
    scale = args.DATASET.DATASET_SCALE
    num_class = args.MODEL.NUM_CLASSES
    cache_mode = args.TRAIN.CACHE_MODE
    ann_file = f'{fold_split}/seed_{seed}/{scale}{image_set}.txt'


    print(f'Build dataset : {scale} with num class {num_class}')
    dataset = WSIDataset(transforms=make_wsi_transforms(image_set),
                         ann_file=ann_file,
                         num_class=num_class,
                         cache_mode=cache_mode,
                         local_rank=get_local_rank(),
                         local_size=get_local_size())

    return dataset

