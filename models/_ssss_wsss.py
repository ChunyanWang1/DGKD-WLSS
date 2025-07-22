import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel
import math
from .sft import SFTLayer
from .mods import PAMR
from .mods import StochasticGate
from .mods import GCI,ASPP
__all__ = ["SSSS"]

#
# Helper classes
#
def rescale_as(x, y, mode="bilinear", align_corners=True):
    h, w = y.size()[2:]
    x = F.interpolate(x, size=[h, w], mode=mode, align_corners=align_corners)
    return x

def focal_loss(x, p = 1, c = 0.1):
    return torch.pow(1 - x, p) * torch.log(c + x)

def pseudo_gtmask(mask, cutoff_top=0.6, cutoff_low=0.2, eps=1e-8):
    """Convert continuous mask into binary mask"""
    bs,c,h,w = mask.size()
    mask = mask.view(bs,c,-1)

    # for each class extract the max confidence
    mask_max, _ = mask.max(-1, keepdim=True)
    mask_max[:, :1] *= 0.7
    mask_max[:, 1:] *= cutoff_top
    #mask_max *= cutoff_top

    # if the top score is too low, ignore it
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)

    # remove ambiguous pixels
    ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
    pseudo_gt = (1 - ambiguous) * pseudo_gt

    return pseudo_gt.view(bs,c,h,w)

def balanced_mask_loss_ce(mask, pseudo_gt, gt_labels, ignore_index=255):
    """Class-balanced CE loss
    - cancel loss if only one class in pseudo_gt
    - weight loss equally between classes
    """

    mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)
    
    # indices of the max classes
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    ignore_mask = pseudo_gt.sum(1) < 1.
    mask_gt[ignore_mask] = ignore_index

    # class weight balances the loss w.r.t. number of pixels
    # because we are equally interested in all classes
    bs,c,h,w = pseudo_gt.size()
    num_pixels_per_class = pseudo_gt.view(bs,c,-1).sum(-1)
    num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)
    class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)
    class_weight = (pseudo_gt * class_weight[:,:,None,None]).sum(1).view(bs, -1)

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index, reduction="none")
    loss = loss.view(bs, -1)

    # we will have the loss only for batch indices
    # which have all classes in pseudo mask
    gt_num_labels = gt_labels.sum(-1).type_as(loss) + 1 # + BG
    ps_num_labels = (num_pixels_per_class > 0).type_as(loss).sum(-1)
    batch_weight = (gt_num_labels == ps_num_labels).type_as(loss)

    loss = batch_weight * (class_weight * loss).mean(-1)
    return loss


class SSSS(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class SSSSHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36],aux=False,**kwargs):
        super(SSSSHead, self).__init__()

        self.aux=aux
        self.aspp = ASPP(4096, 8,nn.BatchNorm2d)
    
        
        self.feat = nn.Sequential(
                nn.Conv2d(256, 256, 3,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
                #nn.AdaptiveAvgPool2d(1)
            )
        
        self.feat2 = nn.Sequential(
                nn.Conv2d(4096, 256, 3,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
                #nn.AdaptiveAvgPool2d(1)
            )
        
        
        self._init_weight()
        self._init_decoder(num_classes)
    
    def _init_decoder(self, num_classes):

        self._aff = PAMR(10,[1, 2, 4, 8, 12, 24])#self.cfg.PAMR_ITER, self.cfg.PAMR_KERNEL

        def conv2d(*args, **kwargs):
            conv = nn.Conv2d(*args, **kwargs)
            torch.nn.init.kaiming_normal_(conv.weight)
            return conv

        def bnorm(*args, **kwargs):
            bn = nn.BatchNorm2d(*args, **kwargs)
            if not bn.weight is None:
                bn.weight.data.fill_(1)
                bn.bias.data.zero_()
            return bn

        # pre-processing for shallow features
        self.shallow_mask = GCI()#self.NormLayer

        # Stochastic Gate
        self.sg = StochasticGate()
        self.fc8_skip = nn.Sequential(conv2d(256, 48, 1, bias=False), bnorm(48), nn.ReLU())
        self.fc8_x = nn.Sequential(conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    bnorm(256), nn.ReLU())

        # decoder
        self.last_conv = nn.Sequential(conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        bnorm(256), nn.ReLU(),
                                        nn.Dropout(0.5),
                                        conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        bnorm(256), nn.ReLU(),
                                        nn.Dropout(0.1),
                                        conv2d(256, num_classes - 1, kernel_size=1, stride=1))

    def run_pamr(self, im, mask):
        im = F.interpolate(im, mask.size()[-2:], mode="bilinear", align_corners=True)
        masks_dec = self._aff(im, mask)
        return masks_dec

    def forward(self,feature,y,labels):
        test_mode= labels is None
        feature['out']=F.relu(feature['out'])

        x = self.aspp(feature['out'])

        x_feat_after_aspp=self.feat2(feature['out'])
        feat_auxout=self.feat(feature['aux_out'])
        

        # 3.1 skip connection for deep features
        x2_x = self.fc8_skip(feature['aux_out'])
        x_up = rescale_as(x, x2_x)
        x = self.fc8_x(torch.cat([x_up, x2_x], 1))

        # 3.2 deep feature context for shallow features
        x2 = self.shallow_mask(feature['aux_out'], x)

        # 3.3 stochastically merging the masks
        x = self.sg(x, x2, alpha_rate=0.3)#self.cfg.SG_PSI


        # 4. final convs to get the masks
        x = self.last_conv(x)

        #
        # 5. Finalising the masks and scores
        #

        # constant BG scores
        bg = torch.ones_like(x[:, :1])
        x = torch.cat([bg, x], 1)

        bs, c, h, w = x.size()

        masks = F.softmax(x, dim=1)

        # reshaping
        features = x.view(bs, c, -1)
        masks_ = masks.view(bs, c, -1)

        # classification loss
        cls_1 = (features * masks_).sum(-1) / (1.0 + masks_.sum(-1))

        # focal penalty loss
        cls_2 = focal_loss(masks_.mean(-1), p=3, c=0.3)#c=0.01 0.3

        # adding the losses together
        cls = cls_1[:, 1:] + cls_2[:, 1:]

        if test_mode:
            # if in test mode, not mask
            # cleaning is performed
            return [cls, None,masks,x_feat_after_aspp]

        self._mask_logits = x

        # # foreground stats
        # masks_ = masks_[:, 1:]
        # cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)

        # mask refinement with PAMR
        masks_dec = self.run_pamr(y, masks.detach())

        # upscale the masks & clean
        masks = self._rescale_and_clean(masks, y, labels)
        masks_dec = self._rescale_and_clean(masks_dec, y, labels)

        # create pseudo GT
        pseudo_gt = pseudo_gtmask(masks_dec).detach()
        loss_mask = balanced_mask_loss_ce(self._mask_logits, pseudo_gt, labels)

        return [cls,loss_mask,masks,x_feat_after_aspp,feat_auxout]
        #return cls, cls_fg, {"cam": masks, "dec": masks_dec}, self._mask_logits, pseudo_gt, loss_mask

    def _rescale_and_clean(self, masks, image, labels):
        """Rescale to fit the image size and remove any masks
        of labels that are not present"""
        masks = F.interpolate(masks, size=image.size()[-2:], mode='bilinear', align_corners=True)
        masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
        return masks


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)






