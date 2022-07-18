import torch.utils.data
from .torchvision_datasets import CocoDetection

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco

def build_dataset(input_data_file, image_set, args):

    dataset_name = args.DATASET.DATASET_NAME

    if dataset_name in ('cnn'):
        from .wsi_feat_dataset import build as build_wsi_feat_dataset
        return build_wsi_feat_dataset(input_data_file, image_set, args)

    raise ValueError(f'dataset {dataset_name} not supported')
