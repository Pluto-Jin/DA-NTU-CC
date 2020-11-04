import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from .NTU import NTU
from .setting import cfg_data 
import torch
import random



def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    train_main_transform = own_transforms.Compose([
        # own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
    	own_transforms.RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    if cfg_data.VAL_MODE=='normal':
        test_list = 'Train Test Splitting list/normal_training/NTU_test_correct.txt'
        train_list = 'Train Test Splitting list/normal_training/NTU_train_correct.txt'
    elif cfg_data.VAL_MODE=='density':
        test_list = 'density_ab+ssc/NTU_density_test_ssc+ab.txt'
        train_list = 'density_ab+ssc/NTU_density_train_ssc+ab.txt'
    elif cfg_data.VAL_MODE=='normal_ab_only':
        test_list = 'normal_ab_only/NTU_test_ab_only.txt'
        train_list = 'normal_ab_only/NTU_train_ab_only.txt'
    elif cfg_data.VAL_MODE=='normal_ssc_only':
        test_list = 'normal_ssc_only/NTU_test_ssc_only.txt'
        train_list = 'normal_ssc_only/NTU_train_ssc_only.txt'
    elif cfg_data.VAL_MODE=='density_ssc_only':
        test_list = 'density_ssc_only/NTU_density_test_ssc_only.txt'
        train_list = 'density_ssc_only/NTU_density_train_ssc_only.txt'
    elif cfg_data.VAL_MODE=='density_ab_only':
        test_list = 'density_ab_only/NTU_density_test_ab_only.txt'
        train_list = 'density_ab_only/NTU_density_train_ab_only.txt'
    elif cfg_data.VAL_MODE=='normal_train_ab_test_ssc':
        test_list = 'normal_train_ab_test_ssc/NTU_test_ssc_correct.txt'
        train_list = 'normal_train_ab_test_ssc/NTU_train_ab_correct.txt'
    elif cfg_data.VAL_MODE=='normal_train_ssc_test_ab':
        test_list = 'normal_train_ssc_test_ab/NTU_test_ab_correct.txt'
        train_list = 'normal_train_ssc_test_ab/NTU_train_ssc_correct.txt'
    elif cfg_data.VAL_MODE=='density_train_ssc_test_ab':
        test_list = 'density_train_ssc_test_ab/NTU_density_split_test_ab_correct.txt'
        train_list = 'density_train_ssc_test_ab/NTU_density_split_train_ssc_correct.txt'
    elif cfg_data.VAL_MODE=='density_train_ab_test_ssc':
        test_list = 'density_train_ab_test_ssc/NTU_density_split_test_ssc_correct.txt'
        train_list = 'density_train_ab_test_ssc/NTU_density_split_train_ab_correct.txt'

    elif cfg_data.VAL_MODE=='hall':
        test_list = 'new_split_list/test.txt'
        train_list = 'new_split_list/train.txt'

    elif cfg_data.VAL_MODE=='hall_DA':
        test_list = 'new_split_list/test.txt'
        train_list = 'Train Test Splitting list/normal_training/NTU_train_correct.txt'
        train_target_list = 'new_split_list/train.txt'

    train_set = NTU(cfg_data.DATA_PATH + train_list, 'train',main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=True)

    val_set = NTU(cfg_data.DATA_PATH + test_list, 'test', main_transform=None, img_transform=img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    if (cfg_data.VAL_MODE=='hall_DA'):
        train_target_set = NTU(cfg_data.DATA_PATH + train_target_list, 'train', main_transform=None, img_transform=img_transform,
                      gt_transform=gt_transform)
        train_target_loader = DataLoader(train_target_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8, shuffle=True,
                                drop_last=True)
        print('source domain:',train_list)
        print('target domain:',train_target_list)
        return train_loader, train_target_loader, val_loader, restore_transform

    return train_loader, val_loader, restore_transform
