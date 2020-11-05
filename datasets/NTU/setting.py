from easydict import EasyDict as edict

# init
__C_NTU = edict()

cfg_data = __C_NTU

__C_NTU.STD_SIZE = (544,960)
# __C_NTU.TRAIN_SIZE = (272,480)
__C_NTU.TRAIN_SIZE = (480,848)#NTU

__C_NTU.DATA_PATH = '/home/jinc0008/dataset/CrowdCounting/'
# __C_NTU.DATA_PATH = '../NTU'

__C_NTU.VAL_MODE = 'hall_DA' # Options:normal,density,normal_ab_only,normal_ssc_only,density_ssc_only,density_ab_only,normal_train_ab_test_ssc,normal_train_ssc_test_ab,density_train_ssc_test_ab,density_train_ab_test_ssc


__C_NTU.DATA_GT = 'k15_s4'            

__C_NTU.MEAN_STD = ([0.40088356,0.40479671,0.37334814], [0.21536005,0.20919993,0.22569714])


__C_NTU.LABEL_FACTOR = 1
# __C_NTU.LOG_PARA = 1000.
__C_NTU.LOG_PARA = 100.

__C_NTU.RESUME_MODEL = ''#model path
# __C_NTU.TRAIN_BATCH_SIZE = 16 #imgs
__C_NTU.TRAIN_BATCH_SIZE = 6 #imgs

# __C_NTU.VAL_BATCH_SIZE = 16 #
__C_NTU.VAL_BATCH_SIZE = 32 #


