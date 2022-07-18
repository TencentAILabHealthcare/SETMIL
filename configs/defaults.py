from yacs.config import CfgNode as CN
import os.path as osp

_C = CN()
_C.TASK = ''
_C.DATA_TABLE = ''
_C.FOLD_SPLIT = ''
_C.NUM_FOLD = 5
_C.LOG_FILE = ''
_C.LABEL_NAME = "Label"
_C.MULTI_LABEL = False

# PATCH
_C.PATCH = CN()
_C.PATCH.IDX2IMG = ''

# PRETRAIN
_C.PRETRAIN = CN()
_C.PRETRAIN.MODEL_NAME = ''
_C.PRETRAIN.MODEL_PATH = ''
_C.PRETRAIN.SAVE_DIR = ''
_C.PRETRAIN.SAVE_DIR_COMBINED = ''
_C.PRETRAIN.TRAIN_FEA_TIMES = 5
_C.PRETRAIN.NUM_CLASSES = 2

# MODEL
_C.MODEL = CN()
_C.MODEL.MODEL_NAME = ''

# # for transformer
_C.MODEL.POSITION_EMBEDDING = ''
_C.MODEL.HIDDEN_DIM = 1
_C.MODEL.NUM_INPUT_CHANNELS = 1
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.LR_BACKBONE_NAME = [] # ["backbone.0"]
_C.MODEL.LR_LINEAR_PROJ_NAME = [] # ['reference_points', 'sampling_offsets']
_C.MODEL.N_HEADS = 8
_C.MODEL.ENCODER_LAYERS = 4
_C.MODEL.DECODER_LAYERS = 2
_C.MODEL.DIM_FEEDFORWARD = 2
_C.MODEL.DROPOUT = 0.5
_C.MODEL.NUM_FEATURE_LEVELS = 1
_C.MODEL.DEC_N_POINTS = 4
_C.MODEL.ENC_N_POINTS = 4
_C.MODEL.TWO_STAGES = False
_C.MODEL.NUM_QUERIES = 100
_C.MODEL.DROP_ENCODER = False
_C.MODEL.OUT_CHENNELS = 256
_C.MODEL.BACKBONE = 'resnet18'
_C.MODEL.DILATION = False
_C.MODEL.MASKS = False
_C.MODEL.AUX_LOSS = False
_C.MODEL.WITH_BOX_REFINE = False

# T2T
_C.MODEL_T2T = CN()
_C.MODEL_T2T.NUM_INPUT_CHANNELS = 1280
_C.MODEL_T2T.NUM_CLASSES = 2
_C.MODEL_T2T.CHANNEL_REDUCE_RATE = 5
_C.MODEL_T2T.TOKEN_DIM = 32
_C.MODEL_T2T.EMBED_DIM = 32
_C.MODEL_T2T.DEPTH = 6
_C.MODEL_T2T.NUM_HEADS = 4
_C.MODEL_T2T.DROP_RATE = 0.1
_C.MODEL_T2T.ATTN_DROP_RATE = 0.1
_C.MODEL_T2T.IRPE = 0
_C.MODEL_T2T.IRPE_METHOD = "euc" # euc, quant, cross, product
_C.MODEL_T2T.IRPE_MODE = "bias" # bias, contextual
_C.MODEL_T2T.IRPE_SHARE_HEAD = True
_C.MODEL_T2T.IRPE_RPE_ON = "q" # q, k, v, [qk, qkv]
_C.MODEL_T2T.ASPP_FLAG = 1
_C.MODEL_T2T.TOKEN_T_IRPE = 0
_C.MODEL_T2T.TOKEN_T_DROP_RATE = 0.1
_C.MODEL_T2T.TOKEN_T_ATTN_DROP_RATE = 0.1
_C.MODEL_T2T.TOKEN_T_IRPE_METHOD = "euc"
_C.MODEL_T2T.TOKEN_T_IRPE_MODE = "bias"
_C.MODEL_T2T.TOKEN_T_IRPE_SHARE_HEAD = True
_C.MODEL_T2T.TOKEN_T_IRPE_RPE_ON = "q"

# DATASET
_C.DATASET = CN()
_C.DATASET.DATASET_NAME = ''
_C.DATASET.DATASET_SEED = 1
_C.DATASET.DATASET_SCALE = 'x20'
_C.DATASET.PATCH_SCORE_PATH = './'
_C.DATASET.PATCH_TOP_K_METHOD = 'random' # random score, tissue, patch_level_pretrained, contrastive_learning
_C.DATASET.FEATURE_LEN = 150 # max feature length in transformer-based dataset
_C.DATASET.FEATURE_MAP_SIZE = 60 # max height,width size of feature map in cnn-dataset
_C.DATASET.TABLE_DATA = './table.csv'
_C.DATASET.FEATURE_NAMES = []
_C.DATASET.TABULAR_NUM= 21


# TRAIN
_C.TRAIN = CN()
_C.TRAIN.SEED = 1
_C.TRAIN.MAX_PATIENCE = 20
_C.TRAIN.EPOCHS = 10000
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.NUM_WORKERS = 8
_C.TRAIN.LR = 1e-3
_C.TRAIN.LR_BACKBONE = 1e-3
_C.TRAIN.LR_LINEAR_PROJ_MULT = 0.1
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.LR_DROP = 40
_C.TRAIN.CLIP_MAX_NORM = 0.01
_C.TRAIN.OPTIM_NAME = 'sgd'
_C.TRAIN.LOSS_NAME = 'focal'
_C.TRAIN.EVAL = False
_C.TRAIN.RESUME_PATH = ''
_C.TRAIN.OUTPUT_DIR = ''
_C.TRAIN.CACHE_MODE = False
_C.TRAIN.MODEL_PATH = CN()
_C.TRAIN.MODEL_PATH.LUAD_GM = './model'
_C.TRAIN.MODEL_PATH.EC_LNM = './model'
# for transformer
_C.TRAIN.CLS_LOSS_COEF = 2
_C.TRAIN.BOX_LOSS_COEF = 5
_C.TRAIN.GIOU_LOSS_COEF = 2
_C.TRAIN.MASK_LOSS_COEF = 1
_C.TRAIN.DICE_LOSS_COEF = 1
_C.TRAIN.FOCAL_ALPHA = 0.25
# for vilt
_C.TRAIN.TEST_ONLY = False
# for metrix
_C.TRAIN.KAPPA = False

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

def update_default_cfg(cfg):
    """
    Args:
        cfg ([type]): [description]
    """
    cfg.TRAIN.OUTPUT_DIR = osp.join(cfg.TRAIN.OUTPUT_DIR, f"network_result")
    cfg.FOLD_SPLIT =  osp.join(cfg.TRAIN.OUTPUT_DIR, f"fold_split")
