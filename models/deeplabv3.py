"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel

from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3 #for 全监督的语义分割
from .backbone import (
    resnet,
    mobilenetv2,
    hrnetv2,
    xception
)

#from .estimate_illumination import Illumination_Estimator
#from .retinex_network import Retinex_decom
#from .sft import SFTLayer

__all__ = ['get_deeplabv3','get_deeplabv3plus']


def _load_model(name, backbone_name,local_rank, pretrained_backbone,num_classes, aux,output_stride=16,**kwargs):
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        local_rank=local_rank,
        replace_stride_with_dilation=replace_stride_with_dilation)
    

    if backbone_name=='resnet101' or backbone_name=='resnet50':
        inplanes = 2048  
    elif backbone_name=='resnet18':
        inplanes = 512
    
    low_level_planes = 256


    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer3':'aux_out','layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate,aux)
    elif name=='deeplabv3':
        return_layers = {'layer3':'aux_out','layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate,aux)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model=DeepLabV3(backbone,classifier)
    return model



def get_deeplabv3(backbone='resnet50', local_rank=None, pretrained=None, 
                  pretrained_base=True, num_class=21,aux=False, **kwargs):
    #for 全监督的语义分割
    model=_load_model('deeplabv3', backbone_name=backbone,local_rank=local_rank, pretrained_backbone=pretrained_base,num_classes=num_class, aux=aux, **kwargs)
    if pretrained != 'None':
        if local_rank is not None:
            device = torch.device(local_rank)
            old_dict=torch.load(pretrained, map_location=device)['model_state']
            model_dict = model.state_dict()
            old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
            model_dict.update(old_dict)
            model.load_state_dict(model_dict)
        else:#!!!!test
            #device = torch.device(local_rank)
            old_dict=torch.load(pretrained)#, map_location=device
            #old_dict=torch.load(pretrained)['model_state'] #for normal images test
            model_dict = model.state_dict()
            
            old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
            model_dict.update(old_dict)
            model.load_state_dict(model_dict)

        

    return model


def get_deeplabv3plus(backbone='resnet50', local_rank=None, pretrained=None, 
                  pretrained_base=True, num_class=19,aux=False, **kwargs):

    # model = DeepLabV3plus(num_class, backbone=backbone, local_rank=local_rank, pretrained_base=pretrained_base, **kwargs)
    # if pretrained != 'None':
    #     if local_rank is not None:
    #         device = torch.device(local_rank)
    #         model.load_state_dict(torch.load(pretrained, map_location=device))
    #for 全监督的语义分割
    model=_load_model('deeplabv3plus', backbone_name=backbone,local_rank=local_rank, pretrained_backbone=pretrained_base,num_classes=num_class, aux=aux, **kwargs)
    if pretrained != 'None':
        if local_rank is not None:
            device = torch.device(local_rank)
            old_dict=torch.load(pretrained, map_location=device)['model_state']
            model_dict = model.state_dict()
            old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
            model_dict.update(old_dict)
            model.load_state_dict(model_dict)
            # new_dict={}
            # for k,v in model_dict.items():
            #     for i,j in old_dict.items():
            #         if i in k:
            #             new_dict[k]=old_dict.pop(i)
            #             break

            # model_dict.update(new_dict)
            # model.load_state_dict(model_dict)
            
        else:#!!!!test
            #device = torch.device(local_rank)
            old_dict=torch.load(pretrained)#, map_location=device
            #old_dict=torch.load(pretrained)['model_state'] #for normal images test
            model_dict = model.state_dict()

            #print(old_dict)
            
            old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
            model_dict.update(old_dict)
            model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    model = get_deeplabv3()
    img = torch.randn(2, 3, 480, 480)
    output = model(img)
