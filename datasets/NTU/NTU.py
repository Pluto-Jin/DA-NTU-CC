import numpy as np
import os
import torch
from PIL import Image
from scipy import io as sio
from torch.utils import data
import pdb

import pandas as pd

from .setting import cfg_data 

class NTU(data.Dataset):
    def __init__(self, list_file, mode, main_transform=None, img_transform=None, gt_transform=None):

#         self.crowd_level = []
#         self.time = []
#         self.weather = []
        self.file_folder = []
        self.file_name = []
#         self.gt_cnt = []
        
        with open(list_file) as f:
            lines = f.read().splitlines()
        
        for line in lines:
            tmp = line.split(' ')
            if len(tmp) == 1:
                tmp = ['hall'] + tmp
            self.file_folder.append(tmp[0])
            self.file_name.append(tmp[1].split('.')[0])

        self.mode = mode
        self.main_transform = main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.num_samples = len(self.file_folder)

    def __getitem__(self, index):
        img, den = self.read_image_and_gt(index)
      
        if self.main_transform is not None:
            img, den = self.main_transform(img,den) 
        if self.img_transform is not None:
            img = self.img_transform(img)

        
        # den = torch.from_numpy(np.array(den, dtype=np.float32))       
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        
        if self.mode == 'train' or self.mode == 'test':    
            return img, den
        else:
            print('invalid data mode!!!')

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,index):

        img_path = os.path.join(cfg_data.DATA_PATH+self.file_folder[index], 'pngs_544_960', self.file_name[index]+'.png')
#         print('img_path',img_path)
        den_map_path = os.path.join(cfg_data.DATA_PATH+self.file_folder[index], 'csv_den_maps_k15_s4_544_960', self.file_name[index]+'.csv')

        img = Image.open(img_path)

        den_map = pd.read_csv(den_map_path, sep=',',header=None).values

        # den_map = sio.loadmat(den_map_path)['den_map'] 

        den_map = den_map.astype(np.float32, copy=False)

        den_map = Image.fromarray(den_map)
        
        return img, den_map


    def get_num_samples(self):
        return self.num_samples       
            
        
