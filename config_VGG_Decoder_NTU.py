import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035 # random seed,  for reporduction
__C.DATASET = 'NTU' # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE

if __C.DATASET == 'UCF50':# only for UCF50
	from datasets.UCF50.setting import cfg_data
	__C.VAL_INDEX = cfg_data.VAL_INDEX 

if __C.DATASET == 'GCC':# only for GCC
	from datasets.GCC.setting import cfg_data
	__C.VAL_MODE = cfg_data.VAL_MODE 

if __C.DATASET == 'NTU':# only for GCC
	from datasets.NTU.setting import cfg_data
	__C.VAL_MODE = cfg_data.VAL_MODE 

__C.NET = 'VGG_DECODER' # net selection: MCNN, VGG, VGG_DECODER, Res50, CSRNet

__C.PRE_GCC = True # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = '04-VGG_decoder_all_ep_21_mae_37.2_mse_91.2.pth' # path to model

__C.RESUME = False # contine training
__C.RESUME_PATH = './exp/04-25_09-19_SHHB_VGG_1e-05/latest_state.pth' # 

__C.GPU_ID = [0] # sigle gpu: [0], [1] ...; multi gpus: [0,1]

# learning rate settings
__C.LR = 1e-6 # learning rate
__C.LR_DECAY = 1# decay rate
__C.LR_DECAY_START = -1 # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1 # decay frequency
__C.MAX_EPOCH = 50

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on

# __C.LAMBDA_1 = 1e-4# SANet:0.001 CMTL 0.0001


# print 
__C.PRINT_FREQ = 30


now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
			 + '_' + __C.DATASET \
             + '_' + __C.NET \
             + '_' + str(__C.LR)

if __C.DATASET == 'UCF50':
	__C.EXP_NAME += '_' + str(__C.VAL_INDEX)	

if __C.DATASET == 'GCC':
	__C.EXP_NAME += '_' + __C.VAL_MODE	

__C.EXP_PATH = './exp/VGG_Decoder_Original_NTU_normal_train_ssc_test_ab_50' # the path of logs, checkpoints, and current codes
if __C.DATASET == 'NTU':
	__C.EXP_NAME += '_' + __C.VAL_MODE	
if __C.DATASET == 'GCC':
	__C.EXP_NAME += '_with_GCC'	
if not os.path.exists(__C.EXP_PATH):
    os.makedirs(__C.EXP_PATH)

#------------------------------VAL------------------------
__C.VAL_DENSE_START = -1
__C.VAL_FREQ = 5 # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1 #  must be 1 for training images with the different sizes



#================================================================================
#================================================================================
#================================================================================  
