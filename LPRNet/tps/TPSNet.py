import torch.nn as nn

from .transformation import TPS_SpatialTransformerNetwork


class TPS(nn.Module):
    def __init__(self, config):
        super(TPS, self).__init__()
        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=config['num_fiducial'], I_size=(config['imgH'], config['imgW']), I_r_size=(config['imgH'], config['imgW']), I_channel_num=config['input_channel'])

    def forward(self, input):
        """ Transformation stage """
        input = self.Transformation(input)
        return input