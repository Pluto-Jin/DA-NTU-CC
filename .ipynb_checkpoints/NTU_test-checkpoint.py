from matplotlib import pyplot as plt
import time
import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd
from collections import OrderedDict
from misc.utils import *
from models.CC import CrowdCounter
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps,ImageDraw,ImageFont
import glob
import argparse

from config_Resnet50_NTU import cfg

parser = argparse.ArgumentParser(description='NTU data')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--save', default='/home/hewei/ntu_random_test_pred', type=str, metavar='PATH',help='path to save prune model (default: current directory)')
parser.add_argument('--test-dir', default='/export/home/hewei/ntu_random_test/', type=str, metavar='PATH',help='path to save prune model (default: current directory)')
parser.add_argument('--model', default='./exp/Res50_Original_NTU_Correct_50/05-18_03-26_NTU_Res50_1e-06_normal/all_ep_33_mae_0.41_mse_0.67.pth',
                    type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--gpu', default="0,1", type=str,
                    help='gpu')
parser.add_argument('--dataset', default='NTU',
                    help='path to the model (default: none)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
torch.backends.cudnn.benchmark = True

font = ImageFont.truetype("/home/hewei/MONO.ttf", 40)

mean_std=([0.40088356,0.40479671,0.37334814], [0.21536005,0.20919993,0.22569714])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()

net = CrowdCounter(cfg.GPU_ID,cfg.NET)
state_dict=torch.load(args.model)

try:
    net.load_state_dict(state_dict['net'])
except KeyError:    
    net.load_state_dict(state_dict)
net.cuda()
net.eval()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

cm = plt.get_cmap('jet')

print(len([i for i in glob.glob(args.test_dir+'*')]))
file_name=[]
file_folder=[]

for file in glob.glob(args.test_dir+'*'):
    file_folder.append(args.test_dir.split('/')[-2])
    file_name.append(os.path.basename(file).replace('.png',''))
    
for folder,file in zip(file_folder,file_name):
    plt.figure()
    imagename='/home/hewei/'+ folder + '/'+ file + '.png'
    
    img = Image.open(imagename)
    img=img.resize((960,544))
    
    if img.mode == 'L':
        img = img.convert('RGB')

#     img_RGBA = img.convert("RGBA")
#     density_map = density_map.convert("RGBA")
#     new_img = Image.blend(img_RGBA, density_map, 0.15)
#     img=new_img.convert("RGB")
      
    input_img = img_transform(img)
    with torch.no_grad():
        pred_map = net.test_forward(Variable(input_img[None,:,:,:]).cuda())
        
    pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
    pred = np.sum(pred_map)/100.0
    print('pred:',pred)
    pred_map = pred_map/np.max(pred_map+1e-20)
    
    # Apply the colormap like a function to any array:
    colored_image_prediction = cm(pred_map)
    prediction=Image.fromarray((colored_image_prediction[:, :, :3] * 255).astype(np.uint8))
    
    draw = ImageDraw.Draw(prediction)
    draw.text((10,10), "Prediction:{:.2f}".format(pred),font=font, fill=(255,0,0))   
    
    concate_img=get_concat_h(img, prediction)
    save_path=os.path.join(args.save,file+'.png')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    concate_img.save(save_path)
    print('save in',save_path)

