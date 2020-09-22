import torch.nn as nn

from tps.transformation import TPS_SpatialTransformerNetwork
from lprnet.LPRNet import LPRNet
from lprnet.load_data import CHARS


class LPR_TPS(nn.Module):
    def __init__(self, config):
        super(LPR_TPS, self).__init__()
        """ Transformation """
        self.transformation = TPS_SpatialTransformerNetwork(
            F=config['tps']['num_fiducial'], I_size=(config['tps']['imgH'], config['tps']['imgW']), I_r_size=(config['tps']['imgH'], config['tps']['imgW']), I_channel_num=config['tps']['input_channel'])
        """ Feature Extraction """
        self.lpr = LPRNet(lpr_max_len=config['lpr']['lpr_max_len'], phase=config['lpr']['phase_train'], class_num=len(CHARS), dropout_rate=config['lpr']['dropout_rate'])
    
    def forward(self, x):
        """ Transformation stage """
        aligned = self.transformation(x)
        """ Feature Extraction Stage"""
        result = self.lpr(aligned)
        return result