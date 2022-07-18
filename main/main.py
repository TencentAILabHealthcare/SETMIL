# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import os.path as osp
import random
import argparse
import datetime
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import build_dataset
from engine import evaluate
from models import build_SETMIL
import util.misc as utils
from util.loss import FocalLoss
from util.logger import getLog
sys.path.insert(0, osp.abspath('./'))
from configs import get_cfg_defaults, update_default_cfg

#%%
def writh_files_in_a_file(input_folder, exception='*pkl', outputfile=None):
    file_list = glob.glob(osp.join(input_folder, exception))
    L = list()
    for file in file_list:
        Bname = osp.basename(osp.splitext(file)[0])
        L.append(f'{Bname},{file},{1}\n')   
    os.makedirs(osp.dirname(outputfile), exist_ok=True)
    with open(outputfile,"w") as file:
        file.writelines(L) 
    
def save_prediction_to_table(input, output_file):
    print(input)
    output_dict = dict()
    output_dict['img_id'] = input['imd_id']
    N, M=np.shape(input['pred'])
    for i in range(M):
        output_dict['pred_prob_class_{}'.format(i)] = input['pred'][:,i]
    df = pd.DataFrame(output_dict)
    df.to_csv(output_file)
    
def main(input_data_file, cfg, task):

    time_now = datetime.datetime.now()
    unique_comment = f'{cfg.MODEL.MODEL_NAME}_{time_now.month}-{time_now.day}-{time_now.hour}-{time_now.minute}'
    cfg.TRAIN.OUTPUT_DIR = osp.join(cfg.TRAIN.OUTPUT_DIR, unique_comment)
    os.makedirs(cfg.TRAIN.OUTPUT_DIR, exist_ok=True)

    logger = getLog(cfg.TRAIN.OUTPUT_DIR + "/log.txt", screen=True)

    # TODO
    utils.init_distributed_mode(cfg)
    logger.info("git:\n  {}\n".format(utils.get_sha()))

    logger.info(cfg)

    device = torch.device("cuda")

    seed = cfg.TRAIN.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model, criterion, postprocessors = build_model(args)
    model_name = cfg.MODEL.MODEL_NAME
    logger.info(f'Build Model: {model_name}')

    if model_name == 'SETMIL':
        model, criterion = build_SETMIL(cfg)
        model.to(device)
    else:
        logger.info(f'Model name not found, {model_name}')
        raise ValueError(f'Model name not found')

    if cfg.TRAIN.LOSS_NAME == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2, num_classes=cfg.MODEL.NUM_CLASSES, reduction="mean")
    elif cfg.TRAIN.LOSS_NAME == "be":
        criterion = nn.CrossEntropyLoss()

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params:  {n_parameters}')    
    
    dataset_test = build_dataset(input_data_file, image_set='test', args=cfg)
    
    print('-'*30)
    if task=='LUAD_GM':
        model_path = cfg.TRAIN.MODEL_PATH.LUAD_GM
    elif task=='EC_LNM':
        model_path = cfg.TRAIN.MODEL_PATH.EC_LNM 
    print('Model Path: {}'.format(model_path))
    test_model=torch.load(model_path)
    model_without_ddp.load_state_dict(test_model["model"])
    
    if cfg.distributed:
        logger.info("ddp is not implemented")
        raise ModuleNotFoundError(f'ddp is not implemented')
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    collate_func = utils.collate_fn_multi_modal if cfg.DATASET.DATASET_NAME in (
    'vilt', 'vilt_surv', 'unit') else utils.collate_fn

    data_loader_test = DataLoader(dataset_test, 
                                  cfg.TRAIN.BATCH_SIZE, 
                                  sampler=sampler_test,
                                  drop_last=False, 
                                  collate_fn=collate_func, 
                                  num_workers=cfg.TRAIN.NUM_WORKERS,
                                  pin_memory=True
                            )

    output_dir = Path(cfg.TRAIN.OUTPUT_DIR)

    test_result = evaluate(logger,
                           model, 
                           criterion, 
                           data_loader_test, 
                           device,
                           cfg.distributed,
                           display_header="Test",
                           save_path = output_dir,
                           kappa_flag=cfg.TRAIN.KAPPA,
                    )
    print("----------------------------")
    print(test_result)
    print("----------------------------")
    save_prediction_to_table(test_result, osp.join(output_dir, 'prediction_result.csv'))
#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="WSI evaluation script")
    parser.add_argument(
        "--cfg",
        default="./configs/default.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="LUAD_GM",
        help="LUAD_GM or EC_LNM",
        type=str,
    )
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    update_default_cfg(cfg)
    input_data_file = osp.join(cfg.TRAIN.OUTPUT_DIR,'wsi_feature_path.txt')
    writh_files_in_a_file(input_folder=cfg.PRETRAIN.SAVE_DIR_COMBINED, exception='*pkl', outputfile=input_data_file)
    main(input_data_file, cfg, task=args.task)
