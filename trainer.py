import numpy as np
import os
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import math

from models.CC import CrowdCounter
# from config import cfg
# from config_resnet50_finetuning import cfg
# from config_VGG_Decoder_finetuning import cfg
# from config_VGG_Decoder_training import cfg
# from config_VGG_Original import cfg
# from config_Resnet50_GCC import cfg
# from config_VGG_decoder_GCC import cfg
# from config_VGG_Decoder_SHHB import cfg
# from config_Resnet50_GCC_finetuning import cfg
# from config_VGG_Decoder_GCC_finetuning import cfg
# from config_Resnet50_GCC_inducing_CAP import cfg
# from config_Resnet50_NTU import cfg
# from config_Resnet50_NTU_finetune import cfg
# from config_Resnet50_NTU_CAP import cfg
# from config_ResSFCN_NTU import cfg
# from config_ResSFCN_GCC import cfg
# from config_ResSFCN_SHHB import cfg
# from config_Res101_GCC import cfg
# from config_Res101_SHHB import cfg

# from config_VGG_Decoder_NTU import cfg
# from config_Resnet50_SHHB import cfg
# from config_CSRNet_GCC import cfg
# from config_CSRNet_SHHB import cfg
# from config_MCNN_SHHB import cfg
# from config_MCNN_GCC import cfg
# from config_MCNN_IN_GCC import cfg
# from config_MCNN_IN_SHHB import cfg

# from config_CSRNet_SHHB import cfg

# from config_SANet_SHHB import cfg

#------------prepare data loader------------
# data_mode = cfg.DATASET
# if data_mode is 'SHHA':
#     from datasets.SHHA.loading_data import loading_data 
#     from datasets.SHHA.setting import cfg_data 
# elif data_mode is 'SHHB':
#     from datasets.SHHB.loading_data import loading_data 
#     from datasets.SHHB.setting import cfg_data 
# elif data_mode is 'QNRF':
#     from datasets.QNRF.loading_data import loading_data 
#     from datasets.QNRF.setting import cfg_data 
# elif data_mode is 'UCF50':
#     from datasets.UCF50.loading_data import loading_data 
#     from datasets.UCF50.setting import cfg_data 
# elif data_mode is 'WE':
#     from datasets.WE.loading_data import loading_data 
#     from datasets.WE.setting import cfg_data 
# elif data_mode is 'GCC':
#     from datasets.GCC.loading_data import loading_data
#     from datasets.GCC.setting import cfg_data
# elif data_mode is 'Mall':
#     from datasets.Mall.loading_data import loading_data
#     from datasets.Mall.setting import cfg_data
# elif data_mode is 'UCSD':
#     from datasets.UCSD.loading_data import loading_data
#     from datasets.UCSD.setting import cfg_data 
# elif data_mode is 'NTU':
#     from datasets.NTU.loading_data import loading_data
#     from datasets.NTU.setting import cfg_data 



from misc.utils import *
import pdb
from collections import OrderedDict

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        
        i_parts = k.split('.')
        i_parts.insert(1,"module")   
        new_state_dict['.'.join(i_parts[0:])] = v
    return new_state_dict

def convert_state_dict_CCN_Module(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        i_parts = k.split('.')
        i_parts.insert(0,"CCN")
        i_parts.insert(1,"module")   
        new_state_dict['.'.join(i_parts[0:])] = v
    return new_state_dict

def convert_state_dict_gcc(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    """E.g., "CCN.module.features4.0.weight" to "CCN.features4.0.weight" """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        
        i_parts = k.split('.')
        i_parts.pop(1) 
        new_state_dict['.'.join(i_parts[0:])] = v
    return new_state_dict

def convert_state_dict_CNN(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
       +CCN.module
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items(): 
        i_parts = k.split('.')
        i_parts.insert(0,"CCN")   
        new_state_dict['.'.join(i_parts[0:])] = v
    return new_state_dict

class Trainer():
    def __init__(self, dataloader, cfg_data, pwd,cfg):

        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd
        self.cfg=cfg

        self.net_name = cfg.NET

        self.net = CrowdCounter(cfg.GPU_ID,self.net_name).cuda()
        self.num_parameters= sum([param.nelement() for param in self.net.parameters()])
        print('num_parameters:',self.num_parameters)
        self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
#         self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)          

        self.train_record = {'best_mae': 1e20, 'best_mse':1e20, 'best_model_name': '_'}

        self.hparam={'lr': cfg.LR, 'n_epochs': cfg.MAX_EPOCH,'number of parameters':self.num_parameters,'dataset':cfg.DATASET}#,'finetuned':cfg.FINETUNE}
        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 

        self.epoch = 0
        self.i_tb = 0
        
        if cfg.PRE_GCC:
            print('===================Loaded Pretrained GCC================')
            weight=torch.load(cfg.PRE_GCC_MODEL)['net']
#             weight=torch.load(cfg.PRE_GCC_MODEL)
            try:
                self.net.load_state_dict(convert_state_dict_gcc(weight))
            except:    
                self.net.load_state_dict(weight)
#             self.net=torch.nn.DataParallel(self.net, device_ids=cfg.GPU_ID).cuda()
          
                        
        self.train_loader, self.val_loader, self.restore_transform = dataloader()

        if cfg.RESUME:
            print('===================Loaded model to resume================')
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']
        #self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp',self.train_loader, self.val_loader, resume=cfg.RESUME,cfg=cfg)


    def forward(self):
#         print('forward!!')
        # self.validate_V3()
        with open(self.log_txt, 'a') as f:
            f.write(str(self.net) + '\n')
            f.write('num_parameters:'+str(self.num_parameters)+'\n')
            
        for epoch in range(self.epoch,self.cfg.MAX_EPOCH):
            self.epoch = epoch
            if epoch > self.cfg.LR_DECAY_START:
                self.scheduler.step()
                
            # training    
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)

            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )
            self.net.eval()
            
            # validation
            if epoch%self.cfg.VAL_FREQ==0 or epoch>self.cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHB', 'QNRF', 'UCF50','Mall']:
                    self.validate_V1()
                elif self.data_mode is 'WE':
                    self.validate_V2()
                elif self.data_mode is 'GCC':
                    self.validate_V3()
                elif self.data_mode is 'NTU':
                    self.validate_V4()
                self.timer['val time'].toc(average=False)
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )


    def train(self): # training for all datasets
        self.net.train()
     
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            img, gt_map = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            self.optimizer.zero_grad()
            pred_map = self.net(img, gt_map)

            loss = self.net.loss
            

            loss.backward()
            self.optimizer.step()

            if (i + 1) % self.cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)

                print( '[ep %d][it %d][loss %.4f][lr %.6f][%.2fs]' % \
                        (self.epoch + 1, i + 1, loss.item(), self.optimizer.param_groups[0]['lr'], self.timer['iter time'].diff) )
                print( '        [cnt: gt: %.1f pred: %.2f]' % (gt_map[0].sum().data/self.cfg_data.LOG_PARA, pred_map[0].sum().data/self.cfg_data.LOG_PARA) )           

        self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.epoch + 1)

    def validate_V1(self):# validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                

                pred_map = self.net.forward(img, gt_map)
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    
                    losses.update(self.net.loss.item())
                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))
                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)



        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,False,self.log_txt)
        print_summary(self.log_txt,self.epoch,self.exp_name,[mae, mse, loss],self.train_record)

    def validate_V2(self):# validate_V2 for WE

        self.net.eval()

        losses = AverageCategoryMeter(5)
        maes = AverageCategoryMeter(5)

        roi_mask = []
        from datasets.WE.setting import cfg_data 
        from scipy import io as sio
        for val_folder in cfg_data.VAL_FOLDER:

            roi_mask.append(sio.loadmat(os.path.join(cfg_data.DATA_PATH,'test',val_folder + '_roi.mat'))['BW'])
        
        for i_sub,i_loader in enumerate(self.val_loader,0):

            mask = roi_mask[i_sub]
            for vi, data in enumerate(i_loader, 0):
                img, gt_map = data

                with torch.no_grad():
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()

                    pred_map = self.net.forward(img,gt_map)

                    pred_map = pred_map.data.cpu().numpy()
                    gt_map = gt_map.data.cpu().numpy()

                    for i_img in range(pred_map.shape[0]):
                    
                        pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                        gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                        losses.update(self.net.loss.item(),i_sub)
                        maes.update(abs(gt_count-pred_cnt),i_sub)
                    if vi==0:
                        vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = np.average(maes.avg)
        loss = np.average(losses.avg)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mae_s1', maes.avg[0], self.epoch + 1)
        self.writer.add_scalar('mae_s2', maes.avg[1], self.epoch + 1)
        self.writer.add_scalar('mae_s3', maes.avg[2], self.epoch + 1)
        self.writer.add_scalar('mae_s4', maes.avg[3], self.epoch + 1)
        self.writer.add_scalar('mae_s5', maes.avg[4], self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, 0, loss],self.train_record,self.log_txt)
        print_WE_summary(self.log_txt,self.epoch,[mae, 0, loss],self.train_record,maes)
#         self.writer.add_hparams(self.hparam, {'best_mae': mae, 'best_mse':mse})





    def validate_V3(self):# validate_V3 for GCC

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        c_maes = {'level':AverageCategoryMeter(9), 'time':AverageCategoryMeter(8),'weather':AverageCategoryMeter(7)}
        c_mses = {'level':AverageCategoryMeter(9), 'time':AverageCategoryMeter(8),'weather':AverageCategoryMeter(7)}


        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map, attributes_pt = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()


                pred_map = self.net.forward(img, gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    s_mae = abs(gt_count-pred_cnt)
                    s_mse = (gt_count-pred_cnt)*(gt_count-pred_cnt)

                    losses.update(self.net.loss.item())
                    maes.update(s_mae)
                    mses.update(s_mse)   
                    attributes_pt = attributes_pt.squeeze() 
                    c_maes['level'].update(s_mae,attributes_pt[i_img][0])
                    c_mses['level'].update(s_mse,attributes_pt[i_img][0])
                    c_maes['time'].update(s_mae,attributes_pt[i_img][1]/3)
                    c_mses['time'].update(s_mse,attributes_pt[i_img][1]/3)
                    c_maes['weather'].update(s_mae,attributes_pt[i_img][2])
                    c_mses['weather'].update(s_mse,attributes_pt[i_img][2])


                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)


        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,False,self.log_txt)


        print_GCC_summary(self.log_txt,self.epoch,[mae, mse, loss],self.train_record,c_maes,c_mses)

    def validate_V4(self):# validate_V4 for NTU
        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()


                pred_map = self.net.forward(img, gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):

                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    s_mae = abs(gt_count-pred_cnt)
                    s_mse = (gt_count-pred_cnt)*(gt_count-pred_cnt)

                    losses.update(self.net.loss.item())
                    maes.update(s_mae)
                    mses.update(s_mse)   
                      


                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)


        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
                [mae, mse, loss],self.train_record,False,self.log_txt)


        print_NTU_summary(self.log_txt,self.epoch,[mae, mse, loss],self.train_record)

if __name__ == '__main__':
    
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


    

    #------------Prepare Trainer------------
    net = cfg.NET

    if net in ['MCNN', 'AlexNet', 'VGG', 'VGG_DECODER','Res50', 'Res101', 'CSRNet','Res101_SFCN']:
        from trainer import Trainer
    elif net in ['SANet']: 
        from trainer_for_M2TCC import Trainer # double losses but signle output
    elif net in ['CMTL']: 
        from trainer_for_CMTL import Trainer # double losses and double outputs
    elif net in ['PCCNet']:
        from trainer_for_M3T3OCC import Trainer

    #------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(loading_data,cfg_data,pwd)
    # print('ready to forward')
    cc_trainer.forward()
