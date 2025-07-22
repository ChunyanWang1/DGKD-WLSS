import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import os
import cv2

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def data_transform(X):
    return 2 * X - 1.0

def denorm(image):

    if image.dim() == 3:
        assert image.dim() == 3, "Expected image [CxHxW]"
        assert image.size(0) == 3, "Expected RGB image [3xHxW]"

        for t, m, s in zip(image, MEAN, STD):
            t.mul_(s).add_(m)
    elif image.dim() == 4:
        # batch mode
        assert image.size(1) == 3, "Expected RGB image [3xHxW]"

        for t, m, s in zip((0,1,2), MEAN, STD):
            image[:, t, :, :].mul_(s).add_(m)

    return image

from scipy import ndimage
import matplotlib.pyplot as plt

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def denorm_img(image):
    ori_images = image.permute(0, 2, 3, 1)#.cpu().numpy()
    #orig_images = np.zeros_like(img_temp)
    ori_images[:, :, :, 0] = (ori_images[:, :, :, 0] * 0.229 + 0.485)
    ori_images[:, :, :, 1] = (ori_images[:, :, :, 1] * 0.224 + 0.456)
    ori_images[:, :, :, 2] = (ori_images[:, :, :, 2] * 0.225 + 0.406)

    return ori_images.cpu().numpy() #.permute(0,3,1,2).contiguous()


def getAttMap(img, attn_map, blur=True):
    # if blur:
    #     attn_map = ndimage.gaussian_filter(attn_map, 0.02*max(img.shape[:2])) #!!!filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    #attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map=img
    # attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
    #         (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def viz_attn(savepath, img, attn_map, blur=True):
    _, axes = plt.subplots(1, 1)#, figsize=(5, 5)
    # print(img.shape)
    # print(attn_map.shape)
    axes.imshow(getAttMap(img, attn_map, blur))
    axes.axis("off")
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0)
    plt.savefig(savepath)

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

        
    # def forward(self, x,d_x):#全监督
    #     #img=x.clone()
    #     input_shape = x.shape[-2:]
    #     #i_map=self.illuminator(x)
    #     #R,L = self.illuminator(x)
    #     #features = self.backbone(x,None)
    #     #x=self.classifier(features)
    #     # middle_shape1=features['out'].shape[-2:]
    #     # middle_shape2=features['aux_out'].shape[-2:]
        

    #     if d_x is None:# teacher
    #     #     # print(features['out'].size())
    #     #     # print(features['aux_out'].size())
    #     #     # print(s_fea['out'].size())
    #     #     # print(s_fea['aux_out'].size())
    #     #     s_fea['out'] = F.interpolate(s_fea['out'], size=middle_shape1, mode='bilinear', align_corners=False)
    #     #     s_fea['aux_out'] = F.interpolate(s_fea['aux_out'], size=middle_shape2, mode='bilinear', align_corners=False)
    #     #     features['out']=self.feature_fusion(features['out'],s_fea['out'],True)
    #     #     features['aux_out']=self.feature_fusion(features['aux_out'],s_fea['aux_out'],False)
            
    #         features = self.backbone(x,None)
    #         x = self.classifier(features,None)
            
    #     #low_R, low_L = self.retinex(features)
    #     #high_R, high_L = self.retinex(high_fea_down8)
    #     else:
    #         #img=x*i_map+x
    #         #img=L*R+x#data_transform(R*torch.pow(L, 0.2)+x) 
    #         #features = self.backbone(img)

    #         features=self.backbone(x,d_x)
    #         #x = self.classifier(features,d_features)#,student
    #         #y=denorm(x)
    #         x = self.classifier(features,d_x)

    #     x[0] = F.interpolate(x[0], size=input_shape, mode='bilinear', align_corners=False)
    #     x[1] = F.interpolate(x[1], size=input_shape, mode='bilinear', align_corners=False)
    #     return x

    def forward(self,x,d_x,labels):#弱监督
        input_shape = x.shape[-2:]
        y=denorm(x)

        if d_x is None:# teacher
            features = self.backbone(x,None)
            x = self.classifier(features,y,labels)
        else:
            #y=denorm(d_x)
            features=self.backbone(x,d_x)
            x = self.classifier(features,y,labels)

        x[2] = F.interpolate(x[2], size=input_shape, mode='bilinear', align_corners=False)
        return x
        


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        self.hrnet_flag = hrnet_flag

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
        #for name, module in model.named_modules():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers
        self.gap=nn.AdaptiveAvgPool2d(1)

    
    def feature_fusion(self,fea1,fea2):#,module
        att1=torch.sigmoid(fea1)
        att2=torch.sigmoid(fea2)
        guide_att=0.5*(1-att1)*(1-att2)+att1*att2
        new_fea=fea2+guide_att*(fea1+fea2)
        #new_fea=fea1+guide_att*(fea1+fea2)

        return new_fea

    def forward(self, x,y):#,filename
        _,_,H,W=x.size()
        out = OrderedDict()
        for name, module in self.named_children():
            if self.hrnet_flag and name.startswith('transition'): # if using hrnet, you need to take care of transition
                if name == 'transition1': # in transition1, you need to split the module to two streams first
                    x = [trans(x) for trans in module]
                else: # all other transition is just an extra one stream split
                    x.append(module(x[-1]))
            else: # other models (ex:resnet,mobilenet) are convolutions in series.
                if y is not None:
                    if name.startswith('CondNet'):
                        prior=module(y)
                        continue
                    if name.startswith('sft'):
                        y=module(x,prior)
                        x=self.feature_fusion(x,y)
                        continue
                    x=module(x) 
                else:
                    x = module(x)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                if name == 'stage4' and self.hrnet_flag: # In HRNetV2, we upsample and concat all outputs streams together
                    output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
                    x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x = torch.cat([x[0], x1, x2, x3], dim=1)
                    out[out_name] = x
                else:
                    out[out_name] = x
        return out
