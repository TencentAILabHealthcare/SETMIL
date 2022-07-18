
import os
import torch
import util.misc as utils
import util.custom_metrics as custom_metrics
from sklearn import metrics
import numpy as np
import pandas as pd

from util.gpu_gather import GpuGather
from typing import Iterable

@torch.no_grad()
def evaluate(logger, model, criterion, data_loader, device, is_distributed, display_header="Valid", is_last_eval=False, save_path='', kappa_flag=True):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'{display_header}:'

    gpu_gather = GpuGather(is_distributed=is_distributed)
    print_freq = len(data_loader) // 4
    print_freq = max(1, print_freq)

    IMG_id = []

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        try:
            label = torch.tensor([x['label'].cpu().numpy() for x in targets])
        except:
            label = torch.tensor([x['label'] for x in targets])

        IMG_id = [x['pid'] for x in targets]
        
        try:
            outputs = model(samples)
            outputs = torch.nn.Softmax(dim=1)(outputs)
        except Exception as e:
            logger.info(samples['target'])
            raise e
        loss = criterion(outputs, label.cuda())

        loss_dict = {'loss': loss}
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        metric_logger.update(loss=loss_dict_reduced['loss'].item())

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        b = outputs.size(0)
        gpu_gather.update(pred=outputs.detach().cpu().view(b, -1).numpy())
        gpu_gather.update(label=label.cpu().numpy().reshape(-1))

    # gather the stats from all processes
    gpu_gather.synchronize_between_processes()

    pred = gpu_gather.pred
    pred = np.concatenate(pred)

    label = gpu_gather.label
    label = np.concatenate(label)

    label = label.reshape(len(pred), -1)
    eval_result = {}
    eval_result['label'] = label
    eval_result['pred'] = pred
    eval_result['imd_id'] = IMG_id

    return eval_result
