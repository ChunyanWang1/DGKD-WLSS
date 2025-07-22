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
from dataset.dataset_vit import VOC12ClassificationDataset,VOC12ClassificationDatasetMSF

import imageio
import imutils
#from thop import profile

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


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Test With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='toco',
                        help='model name')  
    parser.add_argument('--method', type=str, default='kd',
                        help='method name')  
    parser.add_argument('--backbone', type=str, default='vit_base_patch16_224',#resnet18
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='voc12',#'citys',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/VOC2012',#'./dataset/cityscapes/',  
                        help='dataset directory')
    parser.add_argument('--data-list', type=str, default="./dataset/voc12/val.txt",#'./dataset/list/cityscapes/test.lst',  
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
    parser.add_argument('--pretrained', type=str, default='./work_dirs/dgkd-wlss/',
                        help='pretrained seg model')
    parser.add_argument('--save-dir', default='./runs/logs/',
                        help='Directory for saving predictions')
    parser.add_argument('--save-pred', action='store_true', default=True,#True
                    help='save predictions')

    # validation 
    parser.add_argument('--flip-eval', action='store_true', default=True,
                        help='flip_evaluation')
    parser.add_argument('--scales', default=[1.,1.25], type=float, nargs='+', help='multiple scales')#,0.5,1.5,2.0  #1.,0.5,    #1,0.75 for midas 56.0
    args = parser.parse_args()

    args.aux=True
    return args


class Evaluator(object):
    def __init__(self, args, num_gpus):
        self.args = args
        self.num_gpus = num_gpus
        self.device = torch.device(args.device)

        ignore_label = 255 #-1
        self.val_dataset=VOC12ClassificationDatasetMSF(args.data_list, voc12_root=args.data, num_classes=21,scales=args.scales)


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
                                            pretrained_base=None,
                                            test=True,
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

    def predict_whole(self, net, image,hist_image, tile_size):#enhanced_net,
        cls,prediction,_,_,_,_,_ = net(image.cuda(),hist_image.cuda())
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
                    scaled_probs = self.predict_whole(model, scaled_d_image,scaled_depth_image, tile_size)#enhanced_model,


                    if args.flip_eval:
                        print("flip evaluation")
                        flip_scaled_probs = self.predict_whole(model, torch.flip(scaled_d_image, dims=[3]),torch.flip(scaled_depth_image, dims=[3]), tile_size)
                        scaled_probs = 0.5 * (scaled_probs + torch.flip(flip_scaled_probs, dims=[3]))
                    full_probs += scaled_probs
                    k=k+1
                
                full_probs /= len(scales)
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

            if self.args.save_pred:

                pred = torch.argmax(mask, 1)
                pred = pred.cpu().data.numpy()
                
                predict = pred.squeeze(0)

                dark_img=denorm(d_image[0])*255
                # filepath = os.path.join('pred_voc', os.path.splitext(filename[0])[0] + '.png')
                # imageio.imsave(filepath,predict.astype(np.uint8))
                imageio.imsave(os.path.join(args.outdir, os.path.splitext(filename[0])[0] + '.png'), imutils.encode_cmap2(predict,dark_img[0].astype(np.uint8)).astype(np.uint8))
                
                #imageio.imsave(os.path.join(args.outdir, imutils.encode_cmap2(predict,d_image[0].astype(np.uint8)).astype(np.uint8)))
                print('Save mask to ' + os.path.splitext(filename[0])[0] + '.png' + ' Successfully!')

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
