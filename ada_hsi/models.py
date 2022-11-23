import torch 
import torch.nn as nn
import torch.nn.functional as F

from ada_hsi.unet import Unet
from ada_hsi.spixelnet.archs import SpixelNet
from ada_hsi.utils import ideal_spc


SP_ARCHS = {
    'spixelnet': SpixelNet,
    'unet': Unet,
}



class SPCApative(nn.Module):

    def __init__(self,  model_args, model_name='unet', concat_sideinfo=False, pos_encod=False, slic=False):
        print("Sideinformation with multiplicative operation")
        super(SPCApative, self).__init__()

        self.concat_sideinfo = concat_sideinfo
        self.pos_encod = pos_encod
        self.slic = slic

        if self.slic:
            self.concat_sideinfo = False

        model = SP_ARCHS[model_name]
        

        if concat_sideinfo:
            model_args['n_channels'] = model_args['n_channels'] 

        if slic:
            model_args['n_channels'] = 1

        if pos_encod:
            model_args['n_channels'] = model_args['n_channels'] + 2

        self.unet = model(**model_args)
        self.sensing = ideal_spc
        
   
    def forward(self, inputs):        
        return self.unet(inputs) 
    
    def get_input(self, x, y):

        _inputs = y

        if self.slic:
            _inputs = x

        if self.concat_sideinfo:
            _inputs = _inputs * x
            #_inputs = torch.cat([_inputs, x], dim=1)

        # xy coord meshgrid
        if self.pos_encod:
            xy = self.base_positional_encoding(y.shape)
            _inputs = torch.cat([_inputs, xy], dim=1)
        
        return _inputs


    def get_adaptive_decimation(self, inputs):
        x, D = inputs
        y1 = self.sensing(x, D)
        _inputs = self.get_input(x, y1)
        D2 = self.act( self.unet(_inputs) )
        return D2

    def base_positional_encoding(self, input_shape):
        xy = torch.meshgrid(torch.arange(input_shape[2]), torch.arange(input_shape[3]))
        xy = torch.stack(xy, dim=0).float()
        xy = torch.unsqueeze(xy, dim=0)
        xy = torch.tile(xy, dims=(input_shape[0], 1, 1, 1))
        return xy
