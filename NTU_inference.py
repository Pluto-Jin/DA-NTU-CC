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
import argparse
# from config import cfg
# from config_Resnet50_GCC import cfg

from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps,ImageDraw,ImageFont

torch.backends.cudnn.benchmark = True

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        
        i_parts = k.split('.')
        i_parts.insert(1,"module")   
#         name = k[7:]  # remove `module.`
#         print('.'.join(i_parts[0:]))
        new_state_dict['.'.join(i_parts[0:])] = v
#         break
    return new_state_dict

test_list={'normal_training':'NTU_test_correct.txt',
           'normal_ab_only':'NTU_test_ab_only.txt',
           'normal_ssc_only':'NTU_test_ssc_only.txt',
           'density_ab_only':'NTU_density_test_ab_only.txt',
           'density_ssc_only':'NTU_density_test_ssc_only.txt',
           'normal_train_ssc_test_ab':'NTU_test_ab_correct.txt',
           'normal_train_ab_test_ssc':'NTU_test_ssc_correct.txt',
           'density_train_ssc_test_ab':'NTU_density_split_test_ab_correct.txt',
           'density_train_ab_test_ssc':'NTU_density_split_test_ssc_correct.txt',
           'hall':'test.txt',
           'hall_train':'train.txt'
          }

parser = argparse.ArgumentParser(description='Crowd Counting NTU dataset Inference')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--data', default='/home/jinc0008/dataset/CrowdCounting/', type=str, metavar='PATH',
                    help='path to dataroot (default: current directory)')
parser.add_argument('--save', default='/home/jinc0008/temp/', type=str, metavar='PATH',
                    help='path to save model prediction images (default: current directory)')
parser.add_argument('--model-path', default='./exp/VGG_Decoder_Original_NTU_Correct_50/05-18_01-21_NTU_VGG_DECODER_1e-06_normal/all_ep_6_mae_0.71_mse_1.13.pth',
                    type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--test-mode', default='hall',
                    type=str,help='list images to inference (default: none)')
parser.add_argument('--model-type', default='Resnet50', type=str,
                    help='selected model type')
parser.add_argument('--gpu', default='0', type=str,
                    help='selected gpu')
args = parser.parse_args()

if os.path.exists(args.save):
    print('already exist! exit now')
    exit()
if args.model_type=='VGG_Decoder':
    from config_VGG_Decoder_NTU import cfg
elif args.model_type=='Resnet50':
    from config_Resnet50_NTU import cfg

print(args)
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
#font = ImageFont.truetype("/home/hewei/MONO.ttf", 40)
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

# model_path='04-VGG_decoder_all_ep_21_mae_37.2_mse_91.2.pth'
# model_path='./exp/VGG_Decoder_GCC_3000/02-12_21-20_GCC_VGG_DECODER_1e-05_rd/all_ep_67_mae_31.0_mse_78.9.pth'
# model_path='./exp/VGG_Decoder_GCC_Pretrained_Finetuning/0.4/02-18_11-57_GCC_VGG_DECODER__1e-05_finetuned0.4_rd/all_ep_30_mae_40.7_mse_97.2.pth'
# model_path = './exp/Res50_Original_GCC_Inducing_CAP_0.0001_epochs_100/03-16_23-36_GCC_Res50_cam_lr1e-05_CAP_rd/epoch_17_mae_29.93669934532703_mse_75.04405652371433_state.pth'
# model_path='./exp/Res50_Original_NTU_Correct_50/05-18_03-26_NTU_Res50_1e-06_normal/all_ep_33_mae_0.41_mse_0.67.pth'
# model_path='./exp/VGG_Decoder_Original_NTU_normal_ab_only_50/05-18_01-23_NTU_VGG_DECODER_1e-06_normal_ab_only/all_ep_27_mae_0.70_mse_0.96.pth'
# model_path = './exp/Res50_Original_GCC_Inducing_CAP_0.0001_epochs_100_Finetuning/0.7/03-08_12-37_GCC_Res50__1e-05_finetuned_rd/all_ep_29_mae_32.5_mse_93.2.pth'
# pruned_model_path = './exp/Res50_Original_GCC_Inducing_CAP_0.0001_epochs_100_Pruning/0.7/resnet50_GCC_pruned_0.7.pth.tar'
# pruned_model_path = './exp/VGG_Decoder_GCC_Pretrained_Pruning/0.4/VGG_Decoder_GCC_pruned_0.4.pth.tar'

# model_path='05-ResNet-50_all_ep_35_mae_32.4_mse_76.1.pth'

   
net = CrowdCounter(cfg.GPU_ID,cfg.NET)
# net = CrowdCounter(cfg.GPU_ID,cfg.NET,cfg=torch.load(pruned_model_path)['cfg'])
state_dict=torch.load(args.model_path)

try:
    net.load_state_dict(state_dict['net'])
except KeyError:    
    net.load_state_dict(state_dict)
net.cuda()
net.eval()
sum([param.nelement() for param in net.parameters()])


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
cm = plt.get_cmap('jet')

file_folder=[]
file_name=[]

'''
for file in glob.glob('/export/home/jinc0008/ntu_random_test/*'):
    file_folder.append('ntu_random_test')
    file_name.append(os.path.basename(file).replace('.png',''))
'''
    
with open(os.path.join(args.data,'new_split_list',test_list[args.test_mode])) as f:
    lines = f.readlines()
for line in lines:
    file_folder.append('hall')
    file_name.append(line[:-5])
    
    
    
count=0 
fps=0

maes = AverageMeter()
mses = AverageMeter()

mae_gt_10=AverageMeter()
mae_gt_4_lt_10=AverageMeter()

mae_lt_1=AverageMeter()
mae_gt_1_lt_4=AverageMeter()

mse_gt_10=AverageMeter()
mse_gt_4_lt_10=AverageMeter()

mse_lt_1=AverageMeter()
mse_gt_1_lt_4=AverageMeter()

for folder,file in zip(file_folder,file_name):

    print(count,'/',len(file_folder))
    count+=1

    plt.figure()
    filename_no_ext = file.split('.')[0].split('/')[-1]
    denname = args.data + folder+'/csv_den_maps_k15_s4_544_960/' + file + '.csv'
    imagename=args.data + folder+'/pngs_544_960/' + file + '.png'
    
    
    den = pd.read_csv(denname, sep=',',header=None).values
    den = den.astype(np.float32, copy=False)
    gt = np.sum(Image.fromarray(den))
    print('gt:',gt)
    den = den/np.max(den+1e-20)
    colored_density_map = cm(den)
    density_map=Image.fromarray((colored_density_map[:, :, :3] * 255).astype(np.uint8))
    
    img = Image.open(imagename)
    img=img.resize((960,544))
    if img.mode == 'L':
        img = img.convert('RGB')

    img_RGBA = img.convert("RGBA")
    density_map = density_map.convert("RGBA")
    new_img = Image.blend(img_RGBA, density_map, 0.15)
    
      
    
    input_img = img_transform(img)
    d = ImageDraw.Draw(img)
    d.text((10,10), "Ground Truth:{:.1f}".format(gt), fill=(255,0,0))   
    
    with torch.no_grad():
        start_time=time.time()
        pred_map = net.test_forward(Variable(input_img[None,:,:,:]).cuda())
        elapsed_time = time.time() - start_time
    print('inference time:{}'.format(elapsed_time))
    fps+=(1/elapsed_time)
    pred_map = pred_map.cpu().data.numpy()[0,0,:,:]


    pred = np.sum(pred_map)/100.0
    print('pred:',pred)
    pred_map = pred_map/np.max(pred_map+1e-20)
    

    # Apply the colormap like a function to any array:
    colored_image_prediction = cm(pred_map)
    prediction=Image.fromarray((colored_image_prediction[:, :, :3] * 255).astype(np.uint8))
    draw = ImageDraw.Draw(prediction)
    draw.text((10,10), "Prediction:{:.2f}".format(pred), fill=(255,0,0))   
    
    img=new_img.convert("RGB")
    concate_img=get_concat_h(img, prediction)
    save_path=os.path.join(args.save,folder,'prediction',filename_no_ext+'.png')
    print(save_path)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    concate_img.save(save_path)
    print('save in',save_path)
    
    maes.update(abs(gt-pred))
    mses.update((gt-pred)*(gt-pred))

    if gt >=10:
        mae_gt_10.update(abs(gt-pred))
        mse_gt_10.update((gt-pred)*(gt-pred))
    elif gt >=4 and gt<10:
        mae_gt_4_lt_10.update(abs(gt-pred))
        mse_gt_4_lt_10.update((gt-pred)*(gt-pred))
    elif gt >=2 and gt<4:
        mae_gt_1_lt_4.update(abs(gt-pred))
        mse_gt_1_lt_4.update((gt-pred)*(gt-pred))
    elif gt<=1.0: 
        mae_lt_1.update(abs(gt-pred))
        mse_lt_1.update((gt-pred)*(gt-pred))
    
    print('-Current MAE:{:.2f} -'.format(abs(gt-pred)))
    print('-Current MSE:{:.2f} -'.format((gt-pred)*(gt-pred)))
    print('-Current FPS:{:.2f}'.format(1/elapsed_time))
    
mae = maes.avg
mse = np.sqrt(mses.avg)

mae_gt_10=mae_gt_10.avg
mae_gt_4_lt_10=mae_gt_4_lt_10.avg

mae_lt_1=mae_lt_1.avg
mae_gt_1_lt_4=mae_gt_1_lt_4.avg

mse_gt_10=np.sqrt(mse_gt_10.avg)
mse_gt_4_lt_10=np.sqrt(mse_gt_4_lt_10.avg)

mse_lt_1=np.sqrt(mse_lt_1.avg)
mse_gt_1_lt_4=np.sqrt(mse_gt_1_lt_4.avg)

num_parameters = sum([param.nelement() for param in net.parameters()])

output_str=[]

output_str.append('-args:{} -\n'.format(str(args)))
output_str.append('-Num_parameters:{} -\n'.format(num_parameters))
output_str.append('-Mean MAE:{:.2f} -\n'.format(mae))
output_str.append('-Mean MSE:{:.2f} -\n'.format(mse))
output_str.append('-Mean MAE [0,1]:{:.2f} -\n'.format(mae_lt_1))
output_str.append('-Mean MSE [0,1]:{:.2f} -\n'.format(mse_lt_1))
output_str.append('-Mean MAE [2,4):{:.2f} -\n'.format(mae_gt_1_lt_4))
output_str.append('-Mean MSE [2,4):{:.2f} -\n'.format(mse_gt_1_lt_4))
output_str.append('-Mean MAE [4,10):{:.2f} -\n'.format(mae_gt_4_lt_10))
output_str.append('-Mean MSE [4,10):{:.2f} -\n'.format(mse_gt_4_lt_10))
output_str.append('-Mean MAE [10, ):{:.2f} -\n'.format(mae_gt_10))
output_str.append('-Mean MSE [10, ):{:.2f} -\n'.format(mse_gt_10))
output_str.append('-Mean FPS:{:.2f}s -\n'.format(fps/count))

for string in output_str:
    print(string)
with open(os.path.join(args.save,'results.txt'),'a') as output:
    output.write(''.join(i for i in output_str))
    
