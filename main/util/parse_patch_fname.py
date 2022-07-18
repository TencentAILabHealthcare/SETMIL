
import re
import os.path as osp
from typing import Tuple

def parse_patch_fname_origin(fp: str) -> Tuple[int, int]:
    # d15-08498-tile-r16896-c20480-512x512
    fp = osp.basename(fp).rsplit('.', 1)[0]
    psize = int(fp.rsplit('x', 1)[-1])
    r_loc = int(re.findall(r'(?<=r)\d+', fp)[0])
    c_loc = int(re.findall(r'(?<=c)\d+', fp)[0])

    r_loc = r_loc // psize
    c_loc = c_loc // psize
    return (r_loc, c_loc)


def parse_patch_fname_1(fp: str) -> Tuple[int, int]:
    # d15-08498-tile-r16896-c20480-512x512
    fp = osp.basename(fp).rsplit('.', 1)[0]
    psize = int(fp.rsplit('x', 1)[-1])
    r_loc = int(re.findall(r'(?<=r)\d+', fp)[0])
    c_loc = int(re.findall(r'(?<=c)\d+', fp)[0])

    r_loc = r_loc // psize
    c_loc = c_loc // psize
    return (r_loc, c_loc)

def parse_patch_fname_2(fp: str) -> Tuple[int, int]:
    # 18B0001159D_Block_Region_3_5_26_xini_35472_yini_8051
    try:
        psize = int(fp.rsplit('x', 1)[-1])
    except:
        psize = 512

    r_loc = int(re.findall(r'(?<=xini_)\d+', fp)[0])
    c_loc = int(re.findall(r'(?<=yini_)\d+', fp)[0])

    r_loc = r_loc // psize
    c_loc = c_loc // psize
    return (r_loc, c_loc)


def parse_patch_fname(fp: str) -> Tuple[int, int]:
    # d15-08498-tile-r16896-c20480-512x512
    tmp = fp
    try:
        (r_loc, c_loc) = parse_patch_fname_1(fp)
    except:
        (r_loc, c_loc) = parse_patch_fname_2(fp)
    return (r_loc, c_loc)