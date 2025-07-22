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

# KD_MODULES = {
#     # 'cifar_wrn_40_1': dict(modules=['relu', 'fc'], channels=[64, 100]),
#     # 'cifar_wrn_40_2': dict(modules=['relu', 'fc'], channels=[128, 100]),
#     # 'cifar_resnet56': dict(modules=['layer3', 'fc'], channels=[64, 100]),
#     # 'cifar_resnet20': dict(modules=['layer3', 'fc'], channels=[64, 100]),
#     'resnet50': dict(modules=['layer4', 'fc'], channels=[2048, 21]),
#     'resnet101': dict(modules=['layer4', 'fc'], channels=[2048, 21]),
#     'resnet38': dict(modules=['bn7', 'fc8'], channels=[4096, 21]),
#     # 'tv_resnet34': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
#     # 'tv_resnet18': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
#     # 'resnet18': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
#     # 'tv_mobilenet_v2': dict(modules=['features.18', 'classifier'], channels=[1280, 1000]),
#     # 'nas_model': dict(modules=['features.conv_out', 'classifier'], channels=[1280, 1000]),  # mbv2
#     # 'timm_tf_efficientnet_b0': dict(modules=['conv_head', 'classifier'], channels=[1280, 1000]),
#     # 'mobilenet_v1': dict(modules=['model.13', 'fc'], channels=[1024, 1000]),
#     # 'timm_swin_large_patch4_window7_224': dict(modules=['norm', 'head'], channels=[1536, 1000]),
#     # 'timm_swin_tiny_patch4_window7_224': dict(modules=['norm', 'head'], channels=[768, 1000]),
# }



class DiffKD(nn.Module):
    def __init__(self,channels,kernel_size=3):# student,teacher,student_name,teacher_name,
        super().__init__()
        # # register forward hook
        # # dicts that store distillation outputs of student and teacher
        # self._teacher_out = {}
        # self._student_out = {}

        # student_modules = KD_MODULES[student_name]['modules']
        # student_channels = KD_MODULES[student_name]['channels']
        # teacher_modules = KD_MODULES[teacher_name]['modules']
        # teacher_channels = KD_MODULES[teacher_name]['channels']

        # for student_module, teacher_module in zip(student_modules, teacher_modules):
        #     self._register_forward_hook(student, student_module, teacher=False)
        #     self._register_forward_hook(teacher, teacher_module, teacher=True)
        # self.student_modules = student_modules
        # self.teacher_modules = teacher_modules

        # if channels==9:
        #     kernel_size=1
        self.ddim = DDIM(channels, channels, kernel_size=kernel_size, use_ae=False)#False
        if kernel_size==3:
            self.loss = DIST()
        else:
            #self.loss = MMDLoss()#nn.MSELoss()
            self.loss=DIST()#KLDivergence()


    def forward(self, fs, ft,return_loss=True):#,fs_aux
        fs, ddim_loss, ft, rec_loss = self.ddim(fs, ft)#,fs_aux
        kd_loss = self.loss(fs, ft)
        # ddim_loss = torch.zeros(1, device=kd_loss.device)[0]
        # rec_loss = torch.zeros(1, device=kd_loss.device)[0]
        return ddim_loss, rec_loss, kd_loss
    
    # def forward(self,img, fs, ft,return_loss=True):
    #     if return_loss:
    #         losses=self.new_ddim(img,fs,ft,return_loss)
    #         return losses['loss_mask'],losses['loss_texture'],losses['iou']
    #     else:
    #         new_mask=self.new_ddim(img,fs,fs,return_loss)
    #         return new_mask


    # def _register_forward_hook(self, model, name, teacher=False):
    #     if name == '':
    #         # use the output of model
    #         model.register_forward_hook(partial(self._forward_hook, name=name, teacher=teacher))
    #     else:
    #         module = None
    #         for k, m in model.named_modules():
    #             if k == name:
    #                 module = m
    #                 break
    #         module.register_forward_hook(partial(self._forward_hook, name=name, teacher=teacher))

    # def _forward_hook(self, module, input, output, name, teacher=False):
    #     if teacher:
    #         self._teacher_out[name] = output[0] if len(output) == 1 else output
    #     else:
    #         self._student_out[name] = output[0] if len(output) == 1 else output

    # def _reshape_BCHW(self, x):
    #     """
    #     Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
    #     """
    #     if x.dim() == 2:
    #         x = x.view(x.shape[0], x.shape[1], 1, 1)
    #     elif x.dim() == 3:
    #         # swin [B, N, C]
    #         B, N, C = x.shape
    #         H = W = int(math.sqrt(N))
    #         x = x.transpose(-2, -1).reshape(B, C, H, W)
    #     return x
