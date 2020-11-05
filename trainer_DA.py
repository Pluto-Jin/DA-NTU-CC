import numpy as np
import os
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import math

from models.CC import CrowdCounter
from models.Discriminator import FCDiscriminator
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

# ------------prepare data loader------------
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
        i_parts.insert(1, "module")
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
        i_parts.insert(0, "CCN")
        i_parts.insert(1, "module")
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
        i_parts.insert(0, "CCN")
        new_state_dict['.'.join(i_parts[0:])] = v
    return new_state_dict

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m,'weight'):
            nn.init.normal_(m.weight.data,0.0,0.02)
            if hasattr(m,'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data,0.0)
    return init_fun

def cycle(g):
    while True:
        for i in g:
            yield i


class Trainer():
    def __init__(self, dataloader, cfg_data, pwd, cfg):

        self.cfg_data = cfg_data

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd
        self.cfg = cfg

        self.net_name = cfg.NET

        self.net = CrowdCounter(cfg.GPU_ID, self.net_name, DA=True).cuda()

        self.num_parameters = sum([param.nelement() for param in self.net.parameters()])
        print('num_parameters:', self.num_parameters)
        self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        #         self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': '_'}

        self.hparam = {'lr': cfg.LR, 'n_epochs': cfg.MAX_EPOCH, 'number of parameters': self.num_parameters,
                       'dataset': cfg.DATASET}  # ,'finetuned':cfg.FINETUNE}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.epoch = 0
        self.i_tb = 0

        '''discriminator'''
        if cfg.GAN == 'Vanilla':
            self.bce_loss = torch.nn.BCEWithLogitsLoss()
        elif cfg.GAN == 'LS':
            self.bce_loss = torch.nn.MSELoss()

        if cfg.NET == 'Res50':
            self.channel1, self.channel2 = 1024, 128

        self.D1 = FCDiscriminator(self.channel1, self.bce_loss).cuda()
        self.D2 = FCDiscriminator(self.channel2, self.bce_loss).cuda()
        self.D1.apply(weights_init())
        self.D2.apply(weights_init())

        self.d1_opt = optim.Adam(self.D1.parameters(), lr=self.cfg.D_LR, betas=(0.9, 0.99))
        self.d2_opt = optim.Adam(self.D2.parameters(), lr=self.cfg.D_LR, betas=(0.9, 0.99))

        self.scheduler_D1 = StepLR(self.d1_opt, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)
        self.scheduler_D2 = StepLR(self.d2_opt, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)

        '''loss and lambdas here'''
        self.lambda_adv1 = cfg.LAMBDA_ADV1
        self.lambda_adv2 = cfg.LAMBDA_ADV2


        if cfg.PRE_GCC:
            print('===================Loaded Pretrained GCC================')
            weight = torch.load(cfg.PRE_GCC_MODEL)['net']
            #             weight=torch.load(cfg.PRE_GCC_MODEL)
            try:
                self.net.load_state_dict(convert_state_dict_gcc(weight))
            except:
                self.net.load_state_dict(weight)
        #             self.net=torch.nn.DataParallel(self.net, device_ids=cfg.GPU_ID).cuda()

        '''modify dataloader'''
        self.source_loader, self.target_loader, self.test_loader, self.restore_transform = dataloader()
        self.source_len = len(self.source_loader.dataset)
        self.target_len = len(self.target_loader.dataset)
        print("source:",self.source_len)
        print("target:",self.target_len)
        self.source_loader_iter = cycle(self.source_loader)
        self.target_loader_iter = cycle(self.target_loader)

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
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', self.source_loader, self.test_loader, resume=cfg.RESUME, cfg=cfg)

    def forward(self):
        print('forward!!')
        # self.validate_V3()
        with open(self.log_txt, 'a') as f:
            f.write(str(self.net) + '\n')
            f.write('num_parameters:' + str(self.num_parameters) + '\n')

        for epoch in range(self.epoch, self.cfg.MAX_EPOCH):
            self.epoch = epoch

            # training
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)

            if epoch > self.cfg.LR_DECAY_START:
                self.scheduler.step()
                self.scheduler_D1.step()
                self.scheduler_D2.step()

            print('train time: {:.2f}s'.format(self.timer['train time'].diff))
            print('=' * 20)
            self.net.eval()

            # validation
            if epoch % self.cfg.VAL_FREQ == 0 or epoch > self.cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                if self.data_mode in ['SHHA', 'SHHB', 'QNRF', 'UCF50', 'Mall']:
                    self.validate_V1()
                elif self.data_mode is 'WE':
                    self.validate_V2()
                elif self.data_mode is 'GCC':
                    self.validate_V3()
                elif self.data_mode is 'NTU':
                    self.validate_V4()
                self.validate_train()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))

    def train(self):  # training for all datasets
        self.net.train()

        for i in range(max(len(self.source_loader),len(self.target_loader))):
            torch.cuda.empty_cache()
            self.timer['iter time'].tic()
            img, gt_img = self.source_loader_iter.__next__()
            tar, gt_tar = self.target_loader_iter.__next__()

            img = Variable(img).cuda()
            gt_img = Variable(gt_img).cuda()

            tar = Variable(tar).cuda()
            gt_tar = Variable(gt_tar).cuda()

            #gen loss
            # loss, loss_adv, pred, pred1, pred2, pred_tar, pred_tar1, pred_tar2 = self.gen_update(img,tar,gt_img,gt_tar)
            self.optimizer.zero_grad()

            for param in self.D1.parameters():
                param.requires_grad = False
            for param in self.D2.parameters():
                param.requires_grad = False

            # source
            pred1, pred2, pred = self.net(img, gt_img)
            loss = self.net.loss
            loss.backward()
            loss_adv = None

            loss_d1, loss_d2 = None, None

            # target
            if self.cfg.DIS > 0:
                pred_tar1, pred_tar2, pred_tar = self.net(tar, gt_tar)

                loss_adv = self.D1.cal_loss(pred_tar1, 0) * self.cfg.LAMBDA_ADV1

                if self.cfg.DIS > 1:
                    loss_adv += self.D2.cal_loss(pred_tar2, 0) * self.cfg.LAMBDA_ADV2

                loss_adv.backward()

                #dis loss
                loss_d1, loss_d2 = self.dis_update(pred1,pred2,pred_tar1,pred_tar2)
                self.d1_opt.step()
                self.d2_opt.step()

            self.optimizer.step()

            if (i + 1) % self.cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.writer.add_scalar('loss_adv', loss_adv.item(), self.i_tb)
                self.writer.add_scalar('loss_d1', loss_d1.item(), self.i_tb)
                self.writer.add_scalar('loss_d2', loss_d2.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)

                print('[ep %d][it %d][loss %.4f][loss_adv %.4f][loss_d1 %.4f][loss_d2 %.4f][lr %.8f][%.2fs]' % \
                      (self.epoch + 1, i + 1, loss.item(), loss_adv.item() if loss_adv else 0, loss_d1.item() if loss_d1 else 0, loss_d2.item() if loss_d2 else 0, self.optimizer.param_groups[0]['lr'],
                       self.timer['iter time'].diff))
                print('        [cnt: gt: %.1f pred: %.2f]' % (
                gt_img[0].sum().data / self.cfg_data.LOG_PARA, pred[0].sum().data / self.cfg_data.LOG_PARA))

                if self.cfg.DIS > 0:
                    print('        [tar: gt: %.1f pred: %.2f]' % (
                    gt_tar[0].sum().data / self.cfg_data.LOG_PARA, pred_tar[0].sum().data / self.cfg_data.LOG_PARA))

        self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.epoch + 1)

    def gen_update(self,img,tar,gt_img,gt_tar):
        pass
        # return loss,loss_adv,pred,pred1,pred2,pred_tar,pred_tar1,pred_tar2

    def dis_update(self,pred1,pred2,pred_tar1,pred_tar2):
        self.d1_opt.zero_grad()
        # self.d2_opt.zero_grad()

        for param in self.D1.parameters():
            param.requires_grad = True
        for param in self.D2.parameters():
            param.requires_grad = True

            #source
        pred1 = pred1.detach()
        pred2 = pred2.detach()

        loss_d2 = None

        if self.cfg.DIS > 0 :
            loss_d1 = self.D1.cal_loss(pred1, 0)
            loss_d2 = self.D2.cal_loss(pred2, 0)
            loss_d1.backward()
        if self.cfg.DIS > 1:
            loss_d2.backward()

        loss_D1 = loss_d1
        loss_D2 = loss_d2

        #target
        pred_tar1 = pred_tar1.detach()
        pred_tar2 = pred_tar2.detach()

        if self.cfg.DIS > 0:
            loss_d1 = self.D1.cal_loss(pred_tar1, 1)
            loss_d2 = self.D2.cal_loss(pred_tar2, 1)
            loss_d1.backward()
        if self.cfg.DIS > 1:
            loss_d2.backward()

        loss_D1 += loss_d1
        loss_D2 += loss_d2

        return loss_D1,loss_D2

    def validate_train(self):
        self.net.eval()
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for img, gt_map in self.source_loader:

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                _, _, pred_map = self.net.forward(img, gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = (gt_count - pred_cnt) * (gt_count - pred_cnt)

                    losses.update(self.net.loss.item())
                    maes.update(s_mae)
                    mses.update(s_mse)

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)

        print("test on source domain")
        print_NTU_summary(self.log_txt, self.epoch, [mae, mse, loss], self.train_record)


    def validate_V4(self):  # validate_V4 for NTU
        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(self.test_loader, 0):

            img, gt_map = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()

                _, _, pred_map = self.net.forward(img, gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = (gt_count - pred_cnt) * (gt_count - pred_cnt)

                    losses.update(self.net.loss.item())
                    maes.update(s_mae)
                    mses.update(s_mse)

                if vi == 0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)

        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net, self.optimizer, self.scheduler, self.epoch, self.i_tb, self.exp_path,
                                         self.exp_name, [mae, mse, loss], self.train_record, False, self.log_txt)

        print_NTU_summary(self.log_txt, self.epoch, [mae, mse, loss], self.train_record)


if __name__ == '__main__':

    # ------------prepare enviroment------------
    seed = cfg.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    gpus = cfg.GPU_ID
    if len(gpus) == 1:
        torch.cuda.set_device(gpus[0])

    torch.backends.cudnn.benchmark = True

    # ------------Prepare Trainer------------
    net = cfg.NET

    if net in ['MCNN', 'AlexNet', 'VGG', 'VGG_DECODER', 'Res50', 'Res101', 'CSRNet', 'Res101_SFCN']:
        from trainer import Trainer
    elif net in ['SANet']:
        from trainer_for_M2TCC import Trainer  # double losses but single output
    elif net in ['CMTL']:
        from trainer_for_CMTL import Trainer  # double losses and double outputs
    elif net in ['PCCNet']:
        from trainer_for_M3T3OCC import Trainer

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(loading_data, cfg_data, pwd)
    # print('ready to forward')
    cc_trainer.forward()
