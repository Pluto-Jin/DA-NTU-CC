import os
import numpy as np
import torch
import argparse
from decimal import *

# from config import cfg
# from config_resnet50_finetuning import cfg
# from config_VGG_Decoder_training import cfg
# from config_VGG_Original import cfg
from config_Resnet50_NTU import cfg
# from config_ResSFCN_NTU import cfg
# from config_VGG_Decoder_NTU import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

#------------prepare enviroment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

#------------prepare data loader------------
data_mode = cfg.DATASET
if data_mode is 'SHHA':
    from datasets.SHHA.loading_data import loading_data 
    from datasets.SHHA.setting import cfg_data 
elif data_mode is 'SHHB':
    if args.model in ['MCNN','MCNN_BN']:
        from datasets.SHHB_MCNN.loading_data import loading_data
        from datasets.SHHB_MCNN.setting import cfg_data
    else:
        from datasets.SHHB.loading_data import loading_data 
        from datasets.SHHB.setting import cfg_data 
elif data_mode is 'QNRF':
    from datasets.QNRF.loading_data import loading_data 
    from datasets.QNRF.setting import cfg_data 
elif data_mode is 'UCF50':
    from datasets.UCF50.loading_data import loading_data 
    from datasets.UCF50.setting import cfg_data 
elif data_mode is 'WE':
    from datasets.WE.loading_data import loading_data 
    from datasets.WE.setting import cfg_data 
elif data_mode is 'GCC':
    if args.model in ['CSRNet']:
        from datasets.GCC_CSRNet.loading_data import loading_data
        from datasets.GCC_CSRNet.setting import cfg_data
    else:
        from datasets.GCC.loading_data import loading_data
        from datasets.GCC.setting import cfg_data        
elif data_mode is 'Mall':
    from datasets.Mall.loading_data import loading_data
    from datasets.Mall.setting import cfg_data
elif data_mode is 'UCSD':
    from datasets.UCSD.loading_data import loading_data
    from datasets.UCSD.setting import cfg_data 
elif data_mode is 'NTU':
    from datasets.NTU.loading_data import loading_data
    from datasets.NTU.setting import cfg_data 

#------------Prepare Trainer------------
net = cfg.NET

if net in ['MCNN','MCNN_BN','AlexNet', 'VGG', 'VGG_DECODER', 'VGG_DECODER_BN','Res50','Res101','CSRNet','Res101_SFCN','Res101_SFCN_BN',]:
    from trainer import Trainer
elif net in ['SANet','SANet_BN']: 
    from trainer_for_M2TCC import Trainer # double losses but signle output
elif net in ['CMTL']: 
    from trainer_for_CMTL import Trainer # double losses and double outputs
elif net in ['PCCNet']:
    from trainer_for_M3T3OCC import Trainer

#------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]
cc_trainer = Trainer(loading_data,cfg_data,pwd,cfg)
# print('ready to forward')
cc_trainer.forward()
