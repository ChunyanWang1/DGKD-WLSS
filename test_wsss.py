from __future__ import print_function

import os
import sys
import argparse
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from PIL import Image as PILImage
import numpy as np

from models.model_zoo import get_segmentation_model
from utils.score import SegmentationMetric
from utils.logger import setup_logger
from utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
#from dataset.datasets import CSTestSet
from dataset.datasets import VOC12ClassificationDataset,VOC12ClassificationDatasetMSF # CSTrainValSet,
#from models import dip_origin
import imageio
import imutils
import visualization
#from thop import profile
from scipy import ndimage
import matplotlib.pyplot as plt

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def denorm(image):
    ori_images = image.permute(0, 2, 3, 1)#.cpu().numpy()
    #orig_images = np.zeros_like(img_temp)
    ori_images[:, :, :, 0] = (ori_images[:, :, :, 0] * 0.229 + 0.485)
    ori_images[:, :, :, 1] = (ori_images[:, :, :, 1] * 0.224 + 0.456)
    ori_images[:, :, :, 2] = (ori_images[:, :, :, 2] * 0.225 + 0.406)

    return ori_images.cpu().numpy() #.permute(0,3,1,2).contiguous()

def norm(image):
    ori_images = image.permute(0, 2, 3, 1)#.cpu().numpy()
    #orig_images = np.zeros_like(img_temp)
    ori_images[:, :, :, 0] = (ori_images[:, :, :, 0] - 0.485)/ 0.229
    ori_images[:, :, :, 1] = (ori_images[:, :, :, 1] - 0.456)/ 0.224
    ori_images[:, :, :, 2] = (ori_images[:, :, :, 2] - 0.406)/ 0.225

    return ori_images.permute(0,3,1,2).contiguous()

def grad_cam(model, input_tensor, tile_size,target_layer, target_class=None):
    # 注册钩子
    activations = []
    def hook(module, input, output):
        activations.append(output)
    handle = target_layer.register_forward_hook(hook)

    # 获取输出并反向传播
    outputs= model(input_tensor[0],input_tensor[1],input_tensor[2],input_tensor[3])
    output=outputs[0]
    #print(output.size())
    # if target_class is None:
    #     target_class =output.argmax(dim=1).item()
    model.zero_grad()
    #print(target_class)
    label=torch.nonzero(target_class)
    #print(label[0][0].item())
    output[0, label[0][0].item()].backward()

    # 计算权重
    gradients = target_layer.weight.grad
    #print(gradients.size())
    pooled_gradients = torch.mean(gradients, dim=[1,2, 3]).unsqueeze(0)
    #print(pooled_gradients.size())

    #print(activations[0].size())
    cam = torch.sum(activations[0] * pooled_gradients[..., None, None], dim=1)
    cam = torch.relu(cam)  # ReLU激活
    # cam = cam - cam.min()
    # cam = cam / cam.max()
    cam = torch.clamp(cam, min=0)
    cam=F.interpolate(cam.unsqueeze(1),size=tile_size, mode='bilinear', align_corners=True)

    # 释放钩子
    handle.remove()
    return cam.squeeze().squeeze().detach().cpu().numpy()

def getAttMap(img, attn_map, blur=True):
    # if blur:
    #     attn_map = ndimage.gaussian_filter(attn_map, 0.02*max(img.shape[:2])) #!!!filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    #attn_map=img
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def viz_attn(savepath, img, attn_map, blur=True):
    _, axes = plt.subplots(1, 1)#, figsize=(5, 5)
    print(img.shape)
    print(attn_map.shape)
    axes.imshow(getAttMap(img, attn_map, blur))
    axes.axis("off")
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0)
    plt.savefig(savepath)


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Test With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='ssss',
                        help='model name')  
    parser.add_argument('--method', type=str, default='kd',
                        help='method name')  
    parser.add_argument('--backbone', type=str, default='resnet38',#resnet18
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='voc12',#'citys',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='/opt/data/private/diffkd_segmentation/dataset/VOC2012',#'./dataset/cityscapes/',  
                        help='dataset directory')
    parser.add_argument('--data-list', type=str, default="/opt/data/private/diffkd_segmentation/dataset/voc12/val.txt",#'./dataset/list/cityscapes/test.lst',  
                        help='dataset directory')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    # cuda setting
    parser.add_argument('--gpu-id', type=str, default='0') 
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=None)#0 None
    # checkpoint and log
    parser.add_argument('--pretrained', type=str, default='/opt/data/private/diffkd_segmentation/work_dirs/final_test/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_all_change_feature_fusion/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_change_zeodepths/kd_deeplabv3_resnet38_voc12_best_model.pth', #'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_change_zeodepths/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/final_test/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_all_change_feature_fusion/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_multi_SFT/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/final_test/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_DGKD_55.2/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_all_change_feature_fusion/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_single_DGF2/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_all_change_feature_fusion/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_kd_features_KL/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_all_update_without_depth/kd_deeplabv3_resnet38_voc12_best_model_54.7.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_batch_6_with_DGF/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_all_update/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_DGKD/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_all_mask/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_ablation_two_features/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_DGKD/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_kd_feature_and_map/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_all/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_ablation_kd_feature/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_kd_feature_and_map/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_kd/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_ablation_single_feature/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_batch_6_with_DGF/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6_without_depth/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_batch_6/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_batch_6/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_batch_6/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss2/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_kd_feature_and_map_47.6/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_all_update2_47.0/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_ablation_kd_feature_and_map/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_again/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_with_DGF2/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_with_single_DGF_again2/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_with_single_DGF_again/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_with_depth/deeplabv3_resnet38_voc12_best_model.pth', #'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_with_single_DGF/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_with_single_SFT/deeplabv3_resnet38_voc12_best_model.pth', #'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_with_depth2_31.2/deeplabv3_resnet38_voc12_best_model.pth', #'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_all_update2_47.0/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_DGKD_again_45.8/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_ablation_single_feature_44.3/kd_deeplabv3_resnet38_voc12_best_model.pth', #'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_ablation_kd/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_again/deeplabv3_resnet38_voc12_best_model.pth', #'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_with_depth/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_without_depth/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss3_44.0/kd_deeplabv3_resnet38_voc12_best_model.pth', #'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss3/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_kd/kd_deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_with_depth/deeplabv3_resnet38_voc12_best_model.pth',#'/opt/data/private/diffkd_segmentation/pth/dark_VOC2012_no_noise_extreme_dark_old/diffkd_dark_condition_with_depth_usedepthimg_wsss_fixseed/dist_dv3-r38_dv3_r38_dist2_fea_noenhance/b2_lr0.0025_iter40000_321x321_seed1/test1/kd_deeplabv3_resnet38_voc12_best_model.pth',#work_dirs/diffkd/dist_dv3_r38_dark_baseline_wsss_with_depth/deeplabv3_resnet38_voc12_best_model.pth', #'/opt/data/private/diffkd_segmentation/work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss_without_depth/kd_deeplabv3_resnet38_voc12_best_model.pth',##work_dirs/diffkd/dist_dv3-r38_dv3_r38_dist2_dark_condition_wsss/kd_deeplabv3_resnet38_voc12_best_model_41.0.pth',#dist_dv3-r101_dv3_r50_dist2_dark_condition_batch_6_with_hist_test1/kd_deeplabv3_resnet50_voc12_best_model.pth',#dist_dv3_r101_dark_baseline_lr0.005_batch_6/deeplabv3_resnet101_voc12_best_model.pth',#dist_dv3-r101_dv3_r101_dist2_dark_condition_test_batch_6_with_depth/kd_deeplabv3_resnet101_voc12_best_model.pth',#dist_dv3-r101_dv3_r50_dist2_dark_condition_batch_6_with_hist/kd_deeplabv3_resnet50_voc12_best_model_62.2.pth',#dist_dv3_r50_dark_baseline_lr0.005_batch_6/deeplabv3_resnet50_voc12_best_model.pth',#diffkd/dist_dv3-r101_dv3_r50_dist2_dark_condition_with_hist_without_adapter/kd_deeplabv3_resnet50_voc12_best_model.pth',dist_dv3-r101_dv3_r50_dist2_dark_condition_batch_6_with_hist/kd_deeplabv3_resnet50_voc12_best_model_63.4.pth
                        help='pretrained seg model')
    parser.add_argument('--save-dir', default='./runs/logs/',
                        help='Directory for saving predictions')
    parser.add_argument('--save-pred', action='store_true', default=True,#True
                    help='save predictions')

    # validation 
    parser.add_argument('--flip-eval', action='store_true', default=False,
                        help='flip_evaluation')
    parser.add_argument('--scales', default=[1.,0.5,0.75], type=float, nargs='+', help='multiple scales')#1.,0.5,0.75
    args = parser.parse_args()

    if args.backbone.startswith('resnet'):
        args.aux = True
    elif args.backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')
    return args


class Evaluator(object):
    def __init__(self, args, num_gpus):
        self.args = args
        self.num_gpus = num_gpus
        self.device = torch.device(args.device)

        ignore_label = 255 #-1
        self.val_dataset=VOC12ClassificationDatasetMSF(args.data_list, voc12_root=args.data, scales=args.scales)


        val_sampler = make_data_sampler(self.val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=self.val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, 
                                            backbone=args.backbone,
                                            aux=args.aux, 
                                            pretrained=args.pretrained, 
                                            pretrained_base='None',
                                            local_rank=args.local_rank,
                                            norm_layer=BatchNorm2d).to(self.device)
        

        
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank)
            
        self.model.to(self.device)

        self.metric = SegmentationMetric(self.val_dataset.num_class)


    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        #dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def predict_whole(self, net, image,hist_image, tile_size):
        prediction = net(image,hist_image,None)#,filename
        # #prediction = net(image.cuda(),None,None)
        cls_sigmoid = torch.sigmoid(prediction[0])
        cls_labels = (cls_sigmoid >=0.1)
        #print(cls_sigmoid)
        if isinstance(prediction, tuple) or isinstance(prediction, list):
            prediction = prediction[2]
        # #prediction = interp(prediction)
        prediction[:,1:]=prediction[:,1:]*cls_labels[:, :,None, None]
        prediction =F.interpolate(prediction,size=tile_size, mode='bilinear', align_corners=True)

        return prediction

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))

        
        for i, (image,d_image, depth_image,target, label,filename) in enumerate(self.val_loader):
            target = target.to(self.device)
            

            #N_, C_, H_, W_ = image.size()
            N_,C_, H_, W_ = d_image[0].size()
            tile_size = (H_, W_) ####baseline 39.5
            full_probs = torch.zeros((1, self.val_dataset.num_class, H_, W_)).cuda()

            scales = self.args.scales
            k=0

            with torch.no_grad():
                for scale in scales:
                    scale = float(scale)
                    print("Predicting image scaled by %f" % scale)
                    scaled_d_image = d_image[k].to(self.device)
                    scaled_depth_image = depth_image[k].to(self.device)

                    scaled_probs = self.predict_whole(model, scaled_d_image,scaled_depth_image, tile_size)
                    




                    if args.flip_eval:
                        print("flip evaluation")
                        flip_scaled_probs = self.predict_whole(model, torch.flip(scaled_d_image, dims=[3]),torch.flip(scaled_depth_image, dims=[3]), tile_size)
                        scaled_probs = 0.5 * (scaled_probs + torch.flip(flip_scaled_probs, dims=[3]))
                    full_probs += scaled_probs
                    k=k+1
                
                full_probs /= len(scales)
                #full_probs[:,0]=0.38
                #full_probs[:,0] = torch.pow(1 - torch.max(full_probs[:,1:,:,:], dim=1)[0], 3)#3
                #full_probs[:,0,:,:] =torch.pow(full_probs[:,0,:,:],3)#0.2 #1-torch.max(full_probs[:,1:,:,:],dim=1)[0]#
                mask=full_probs.clone()
                _,_,H,W=full_probs.size()
                #print(full_probs.size())
                target = target.view(H * W)
                eval_index = target !=255
            
                full_probs = full_probs.view(1, self.metric.nclass, H * W)[:,:,eval_index].unsqueeze(-1)
                target = target[eval_index].unsqueeze(0).unsqueeze(-1)
                self.metric.update(full_probs, target)
                mIoU = self.metric.get()
                print("Sample: {:d}, Validation mIoU: {:.3f}".format(i + 1, mIoU))  

            # if self.args.save_pred:
            #     pred = torch.argmax(mask, 1)
            #     pred = pred.cpu().data.numpy()
            #     #seg_pred = self.id2trainId(pred, self.id_to_trainid, reverse=True)
                
            #     predict = pred.squeeze(0)
            #     #dark_img=denorm
            #     #mask = get_color_pallete(predict, self.args.dataset)
            #     #mask = PILImage.fromarray(predict.astype('uint8'))
            #     #mask.save(os.path.join(args.outdir, os.path.splitext(filename[0])[0] + '.png'))
            #     #d_image[0]=F.interpolate(d_image[0],size=tile_size, mode='bilinear', align_corners=True)

            #     dark_img=denorm(d_image[0])*255
            #     filepath = os.path.join('pred_voc', os.path.splitext(filename[0])[0] + '.png')
            #     #imageio.imsave(filepath,predict.astype(np.uint8))
            #     imageio.imsave(os.path.join(args.outdir, os.path.splitext(filename[0])[0] + '.png'), imutils.encode_cmap2(predict,dark_img[0].astype(np.uint8)).astype(np.uint8))
                
            #     #imageio.imsave(os.path.join(args.outdir, imutils.encode_cmap2(predict,d_image[0].astype(np.uint8)).astype(np.uint8)))
            #     print('Save mask to ' + os.path.splitext(filename[0])[0] + '.png' + ' Successfully!')

        synchronize()


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # TODO: optim code
    outdir = '{}_{}_{}_{}'.format(args.model, args.backbone, args.dataset, args.method)
    args.outdir = os.path.join(args.save_dir, outdir)
    if args.save_pred:
        if (args.distributed and args.local_rank == 0) or args.distributed is False:
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)

    logger = setup_logger("semantic_segmentation", args.save_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args, num_gpus)
    evaluator.eval()
    torch.cuda.empty_cache()
