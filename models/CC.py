import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name,DA=False,cfg=None):
        super(CrowdCounter, self).__init__()        

        if model_name == 'AlexNet':
            from .SCC_Model.AlexNet import AlexNet as net        
        elif model_name == 'VGG':
            from .SCC_Model.VGG import VGG as net
        elif model_name == 'VGG_DECODER':
            from .SCC_Model.VGG_decoder import VGG_decoder as net
        elif model_name == 'VGG_DECODER_BN':
            from .SCC_Model.VGG_decoder_bn import VGG_decoder_bn as net
        elif model_name == 'VGG_DECODER_BN_':
            from .SCC_Model.VGG_decoder_bn_ import VGG_decoder_bn as net
        elif model_name == 'VGG_DECODER_BN_cam':
            from .SCC_Model.VGG_decoder_bn_cam import VGG_decoder_ as net
        elif model_name == 'VGG_DECODER_IN':
            from .SCC_Model.VGG_decoder_in import VGG_decoder_in as net
        elif model_name == 'VGG_DECODER_IN_':
            from .SCC_Model.VGG_decoder_in_ import VGG_decoder_in as net
        elif model_name == 'VGG_DECODER_IN_cam':
            from .SCC_Model.VGG_decoder_in_cam import VGG_decoder_in as net
        elif model_name == 'VGG_DECODER_':
            from .SCC_Model.VGG_decoder_ import VGG_decoder_ as net
        elif model_name == 'MCNN':
            from .SCC_Model.MCNN import MCNN as net
        elif model_name == 'MCNN_IN':
            from .SCC_Model.MCNN_IN import MCNN_IN as net
        elif model_name == 'MCNN_BN':
            from .SCC_Model.MCNN_BN import MCNN as net
        elif model_name == 'CSRNet':
            from .SCC_Model.CSRNet import CSRNet as net
        elif model_name == 'CSRNet_IN':
            from .SCC_Model.CSRNet_IN import CSRNet as net
        elif model_name == 'Res50':
            from .SCC_Model.Res50 import Res50 as net
        elif model_name == 'Res50_':
            from .SCC_Model.Res50_ import Res50 as net
        elif model_name == 'Res50_cam':
            from .SCC_Model.Res50_cam import Res50 as net
        elif model_name == 'Res50_IN_cam':
            from .SCC_Model.Res50_IN_cam import Res50 as net
        elif model_name == 'Res50_IN': 
            from .SCC_Model.Res50_IN import Res50 as net
        elif model_name == 'Res50_IN_': 
            from .SCC_Model.Res50_IN_ import Res50 as net
        elif model_name == 'Res50_GN':
            from .SCC_Model.Res50_GN import Res50 as net
        elif model_name == 'Res101':
            from .SCC_Model.Res101 import Res101 as net      
        elif model_name == 'Res101_':
            from .SCC_Model.Res101_ import Res101 as net 
        elif model_name == 'Res101_cam':
            from .SCC_Model.Res101_cam import Res101 as net 
        elif model_name == 'Res101_IN':
            from .SCC_Model.Res101_IN import Res101_IN as net   
        elif model_name == 'Res101_IN_':
            from .SCC_Model.Res101_IN_ import Res101_IN as net   
        elif model_name == 'Res101_IN_cam':
            from .SCC_Model.Res101_IN_cam import Res101_IN as net
        elif model_name == 'Res101_SFCN':
            from .SCC_Model.Res101_SFCN import Res101_SFCN as net
        elif model_name == 'Res101_SFCN_BN':
            from .SCC_Model.Res101_SFCN_BN import Res101_SFCN as net
        elif model_name == 'Res101_SFCN_IN':
            from .SCC_Model.Res101_SFCN_IN import Res101_SFCN_IN as net
        elif model_name == 'Res101_SFCN_BN_':
            from .SCC_Model.Res101_SFCN_BN_ import Res101_SFCN as net
        elif model_name == 'Res101_SFCN_IN_':
            from .SCC_Model.Res101_SFCN_IN_ import Res101_SFCN_IN as net
        elif model_name == 'Res101_SFCN_IN_cam':
            from .SCC_Model.Res101_SFCN_IN_cam import Res101_SFCN_IN as net

        self.DA = DA
        print('============Using {}============='.format(model_name))
        if cfg==None:
            print('cfg is None in CC.py')
            if self.DA:
                print('Domain Adaptation!')
                self.CCN = net(DA=self.DA)
            else:
                print('no Domain Adaptation!')
                self.CCN = net()
        else:
            print('cfg is not None in CC.py')
            self.CCN = net(cfg=cfg)

#         if model_name == 'VGG_DECODER':
#             print('load VGG_DECODER with pretrained!')

#             self.CCN = net(pretrained=True)
#             print(self.CCN.state_dict()['features4.0.weight'])
        if len(gpus)>1:
            print('Using DataParallel gpu {}'.format(gpus))
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN=self.CCN.cuda()
        if model_name in ['Res50_cam','Res50_IN_cam','Res101_IN_cam','Res101_SFCN_IN_cam','VGG_DECODER_IN_cam','VGG_DECODER_BN_cam']:
            print('========== CAP Sparsification mode =====')
        self.loss_mse_fn = nn.MSELoss().cuda()
        self.model_name=model_name
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map):
        if self.model_name in ['Res50_cam','Res50_IN_cam','Res101_IN_cam','Res101_cam','Res101_SFCN_IN_cam','VGG_DECODER_IN_cam','VGG_DECODER_BN_cam']:
            density_map,energy = self.CCN(img)                    
        elif self.DA:
            d1_map, d2_map, density_map = self.CCN(img)
            # print('d1_map.shape',d1_map.shape)
            # print('d2_map.shape',d2_map.shape)
            # print('density_map.shape', density_map.shape)
        else:
            density_map = self.CCN(img)

#         print('gt_map.shape',gt_map.shape)
#         print('density_map.shape',density_map.shape)

        self.loss_mse= self.build_loss(density_map.squeeze(), gt_map.squeeze())           
        if self.model_name in ['Res50_cam','Res50_IN_cam','Res101_IN_cam','Res101_cam','Res101_SFCN_IN_cam','VGG_DECODER_IN_cam','VGG_DECODER_BN_cam']:
            return density_map,energy
        elif self.DA:
            return d1_map, d2_map, density_map
        else:
            return density_map
    
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):
        if self.model_name in ['Res50_cam','Res50_IN_cam','Res101_IN_cam','Res101_cam','Res101_SFCN_IN_cam','VGG_DECODER_IN_cam','VGG_DECODER_BN_cam']:
            density_map,energy = self.CCN(img)                    
            return density_map,energy
        elif self.DA:
            d1_map, d2_map, density_map = self.CCN(img)
            return d1_map, d2_map, density_map
        else:
            density_map = self.CCN(img)                    
            return density_map

