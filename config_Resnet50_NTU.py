import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035 # random seed,  for reproduction
__C.DATASET = 'NTU' # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD

if __C.DATASET == 'UCF50':# only for UCF50
    from datasets.UCF50.setting import cfg_data
    __C.VAL_INDEX = cfg_data.VAL_INDEX 

if __C.DATASET == 'GCC':# only for GCC
    from datasets.GCC.setting import cfg_data
    __C.VAL_MODE = cfg_data.VAL_MODE 
if __C.DATASET == 'NTU':# only for GCC
    from datasets.NTU.setting import cfg_data
    __C.VAL_MODE = cfg_data.VAL_MODE


__C.NET = 'Res50' # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet

#DA settings
__C.DA = True #domain adaptation flag
__C.GAN = 'Vanilla' #Vanilla, LS
__C.LAMBDA_ADV1 = 2e-8 #2e-4(not good),2e-5(not good),2e-6(not good),2e-7,2e-8
__C.LAMBDA_ADV2 = 0.001
__C.DIS = 1
__C.D_LR = 1e-5 #discriminator lr 1e-5(default),1e-6(just so so)


__C.PRE_GCC = False # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = './exp/best_state.pth' # path to model

__C.RESUME = False # contine training
__C.RESUME_PATH = './exp/04-25_09-19_SHHB_VGG_1e-05/latest_state.pth' # 

__C.GPU_ID = [0] # single gpu: [0], [1] ...; multi gpus: [0,1]

# learning rate settings
__C.LR = 2e-5 # learning rate 2e-5(default),1e-5(just so so),1e-6
__C.LR_DECAY = 0.5 # decay rate
__C.LR_DECAY_START = 20 # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 10 # decay frequency
__C.MAX_EPOCH = 50

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on

__C.LAMBDA_1 = 1e-4# SANet:0.001 CMTL 0.0001


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

__C.EXP_PATH = './exp/Res50_NTU_50/' # the path of logs, checkpoints, and current codes
if __C.DATASET == 'NTU':
	__C.EXP_NAME += '_' + __C.VAL_MODE	

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
