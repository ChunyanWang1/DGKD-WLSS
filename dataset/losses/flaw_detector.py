import torch
import torch.nn as nn
import torch.nn.functional as F
from .gaussian_blur import GaussianBlurLayer

class FlawDetector(nn.Module):
    """ The FC Discriminator proposed in paper:
        'Guided Collaborative Training for Pixel-wise Semi-Supervised Learning'
    """

    ndf = 64    # basic number of channels
	
    def __init__(self, in_channels):
        super(FlawDetector, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, self.ndf, kernel_size=4, stride=2, padding=1)
        self.ibn1 = IBNorm(self.ndf)
        self.conv2 = nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1)
        self.ibn2 = IBNorm(self.ndf * 2)
        self.conv2_1 = nn.Conv2d(self.ndf * 2, self.ndf * 2, kernel_size=4, stride=1, padding=1)
        self.ibn2_1 = IBNorm(self.ndf * 2)
        self.conv3 = nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=4, stride=2, padding=1)
        self.ibn3 = IBNorm(self.ndf * 4)
        self.conv3_1 = nn.Conv2d(self.ndf * 4, self.ndf * 4, kernel_size=4, stride=1, padding=1)
        self.ibn3_1 = IBNorm(self.ndf * 4)
        self.conv4 = nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=4, stride=2, padding=1)
        self.ibn4 = IBNorm(self.ndf * 8)
        self.conv4_1 = nn.Conv2d(self.ndf * 8, self.ndf * 8, kernel_size=4, stride=1, padding=1)
        self.ibn4_1 = IBNorm(self.ndf * 8)
        self.classifier = nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, task_inp, task_pred):
        resulter, debugger = {}, {}

        #task_inp = torch.cat(task_inp, dim=1)
        x = torch.cat((task_inp, task_pred), dim=1)
        x = self.leaky_relu(self.ibn1(self.conv1(x)))
        x = self.leaky_relu(self.ibn2(self.conv2(x)))
        x = self.leaky_relu(self.ibn2_1(self.conv2_1(x)))
        x = self.leaky_relu(self.ibn3(self.conv3(x)))
        x = self.leaky_relu(self.ibn3_1(self.conv3_1(x)))
        x = self.leaky_relu(self.ibn4(self.conv4(x)))
        x = self.leaky_relu(self.ibn4_1(self.conv4_1(x)))
        x = self.classifier(x)
        x = F.interpolate(x, size=(task_pred.shape[2], task_pred.shape[3]), mode='bilinear', align_corners=True)

        # x is not activated here since it will be activated by the criterion function
        assert x.shape[2:] == task_pred.shape[2:]
        #resulter['flawmap'] = x
        resulter = x
        return resulter #, debugger


class IBNorm(nn.Module):
    """ This layer combines BatchNorm and InstanceNorm.
    """

    def __init__(self, num_features, split=0.5):
        super(IBNorm, self).__init__()

        self.num_features = num_features
        self.num_BN = int(num_features * split + 0.5)
        self.bnorm = nn.BatchNorm2d(num_features=self.num_BN, affine=True)#SynchronizedBatchNorm2d(num_features=self.num_BN, affine=True)
        self.inorm = nn.InstanceNorm2d(num_features=num_features - self.num_BN, affine=False)

    def forward(self, x):
        if self.num_BN == self.num_features:
            return self.bnorm(x.contiguous())
        else:
            xb = self.bnorm(x[:, 0:self.num_BN, :, :].contiguous())
            xi = self.inorm(x[:, self.num_BN:, :, :].contiguous())

            return torch.cat((xb, xi), 1)


class FlawDetectorCriterion(nn.Module):
    """ Criterion of the flaw detector.
    """

    def __init__(self):
        super(FlawDetectorCriterion, self).__init__()

    def forward(self, pred, gt, is_ssl=False, reduction=True):    
        loss = F.mse_loss(pred, gt, reduction='none')
        if reduction:
            loss = torch.mean(loss, dim=(1, 2, 3))
        return loss
    

class FlawmapHandler(nn.Module):
    """ Post-processing of the predicted flawmap.

    This module processes the predicted flawmap to fix some special 
    cases that may cause errors in the subsequent steps of generating
    pseudo ground truth.
    """
    
    def __init__(self):
        super(FlawmapHandler, self).__init__()
        #self.args = args
        self.clip_threshold = 0.1

        blur_ksize = int(448 / 16)
        blur_ksize = blur_ksize + 1 if blur_ksize % 2 == 0 else blur_ksize
        self.blur = GaussianBlurLayer(1, blur_ksize)

    def forward(self, flawmap):
        flawmap = flawmap.data

        # force all values to be larger than 0
        flawmap.mul_((flawmap >= 0).float())
        # smooth the flawmap
        flawmap = self.blur(flawmap)
        # if all values in the flawmap are less than 'clip_threshold'
        # set the entire flawmap to 0, i.e., no flaw pixel
        fmax = flawmap.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        fmin = flawmap.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        max_matrix = fmax.repeat(1, 1, flawmap.shape[2], flawmap.shape[3])
        flawmap.mul_((max_matrix > self.clip_threshold).float())
        # normalize the flawmap
        flawmap = flawmap.sub_(fmin).div_(fmax - fmin + 1e-9)

        return flawmap


class DCGTGenerator(nn.Module):
    """ Generate the ground truth of the dynamic consistency constraint.
    """

    def __init__(self):
        super(DCGTGenerator, self).__init__()
        #self.args = args
        self.dc_threshold=0.6

    def forward(self, l_pred, r_pred, l_handled_flawmap, r_handled_flawmap):
        l_tmp = l_handled_flawmap.clone()
        r_tmp = r_handled_flawmap.clone()

        l_bad = l_tmp > self.dc_threshold
        r_bad = r_tmp > self.dc_threshold

        both_bad = (l_bad & r_bad).float()

        l_handled_flawmap.mul_((l_tmp <= self.dc_threshold).float())
        r_handled_flawmap.mul_((r_tmp <= self.dc_threshold).float())

        l_handled_flawmap.add_((l_tmp > self.dc_threshold).float())
        r_handled_flawmap.add_((r_tmp > self.dc_threshold).float())

        l_mask = (r_handled_flawmap >= l_handled_flawmap).float()
        r_mask = (l_handled_flawmap >= r_handled_flawmap).float()

        l_dc_gt = l_mask * l_pred + (1 - l_mask) * r_pred
        r_dc_gt = r_mask * r_pred + (1 - r_mask) * l_pred

        return l_dc_gt, r_dc_gt, both_bad, both_bad


class FDGTGenerator(nn.Module):
    """ Generate the ground truth of the flaw detector, 
        i.e., pipeline 'C' in the paper.
    """

    def __init__(self):
        super(FDGTGenerator, self).__init__()
        #self.args = args
        self.mu=0.5
        self.nu=1

        blur_ksize = int(448 / 8)
        blur_ksize = blur_ksize + 1 if blur_ksize % 2 == 0 else blur_ksize
        self.blur = GaussianBlurLayer(1, blur_ksize)

        reblur_ksize = int(448 / 4)
        reblur_ksize = reblur_ksize + 1 if reblur_ksize % 2 == 0 else reblur_ksize
        self.reblur = GaussianBlurLayer(1, reblur_ksize)

        self.dilate = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        )

    def forward(self, pred, gt):
        diff = torch.abs_(gt - pred.detach())
        diff = torch.sum(diff, dim=1, keepdim=True).mul_(self.mu)
        
        diff = self.blur(diff)
        for _ in range(0, self.nu):
            diff = self.reblur(self.dilate(diff))

        # normlize each sample to [0, 1]
        dmax = diff.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        dmin = diff.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        diff.sub_(dmin).div_(dmax - dmin + 1e-9)

        flawmap_gt = diff
        return flawmap_gt
