import torch
import torch.nn as nn
from .ddim import DDIM
from .loss import CriterionKD
from .dist_kd import DIST
from .kl_div import KLDivergence
#from .new_ddim import SegRefiner
from .mmdloss import MMDLoss
from functools import partial
import math



class DiffKD(nn.Module):
    def __init__(self,channels,kernel_size=3):
        super().__init__()
        self.ddim = DDIM(channels, channels, kernel_size=kernel_size)
        self.loss=DIST()


    def forward(self, fs, ft,return_loss=True):#,fs_aux
        fs, ddim_loss, ft = self.ddim(fs, ft)#,fs_aux
        kd_loss = self.loss(fs, ft)
        return ddim_loss, kd_loss