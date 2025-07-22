"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .toco_model.model_seg_neg import network


def get_toco(backbone='vit_base_patch16_224', local_rank=None, pretrained=None, pretrained_base=True, num_class=21,test=False, **kwargs):
    if pretrained==None and pretrained_base==True:
        s_model=True
    else:
        s_model=False
    model = network(
        backbone=backbone,
        num_classes=num_class,
        pretrained=pretrained_base,
        init_momentum=0.9,
        aux_layer=-3,
        s_model=s_model
    )

    
    if pretrained is not None:
        trained_state_dict = torch.load(pretrained, map_location="cpu")
        model_dict = model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in trained_state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
        # new_state_dict.pop("conv.weight")
        # new_state_dict.pop("aux_conv.weight")
        model.load_state_dict(state_dict=new_state_dict, strict=False)
    elif test:
        old_dict=torch.load(pretrained)#, map_location=device
        #old_dict=torch.load(pretrained)['model_state'] #for normal images test
        model_dict = model.state_dict()

        #print(old_dict)
        
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    
    return model

