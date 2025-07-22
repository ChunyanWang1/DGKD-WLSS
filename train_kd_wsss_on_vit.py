import argparse
import time
import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import shutil
import sys
import math

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

import random
import numpy as np


from losses.loss import SegCrossEntropyLoss, CriterionKD, CriterionMiniBatchCrossImagePair
from losses.diffkd import DiffKD
from models.model_zoo import get_segmentation_model
from models.PAR import PAR

from utils.distributed import *
from utils.logger import setup_logger
from utils.score import SegmentationMetric
from dataset.dataset_vit import VOC12ClassificationDataset,VOC12ClassificationDatasetMSF # CSTrainValSet,

from utils.flops import cal_multi_adds, cal_param_size#,PolynomialLR
#from models.retinex_network import Retinex_decom
# from models import dip_origin

from losses.dist_kd import DIST
from losses.kl_div import KLDivergence
# from losses.mmdloss import MMDLoss

# from utils.cutmix import obtain_bbox,mix

#from losses.flaw_detector import FlawDetector,FDGTGenerator,FlawDetectorCriterion,FlawmapHandler,DCGTGenerator
from torch.cuda.amp import autocast, GradScaler

from camutils import cam_to_label, cam_to_roi_mask2, multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2, crop_from_roi_neg
from losses.losses import  get_seg_loss #, get_masked_ptc_loss,CTCLoss_neg, DenseEnergyLoss, get_energy_loss
import optimizer
# from depth_estimator import networks
# from depth_estimator.layers import disp_to_depth
# # import cv2
# # import heapq
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# from thop import profile
# from ptflops.flops_counter import get_model_complexity_info

loss_mse = torch.nn.MSELoss()    
# Calculate Gram matrix (G = FF^T)
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

def denorm(image):
    ori_images = image.permute(0, 2, 3, 1)#.cpu().numpy()
    #orig_images = np.zeros_like(img_temp)
    ori_images[:, :, :, 0] = (ori_images[:, :, :, 0] * 0.229 + 0.485)
    ori_images[:, :, :, 1] = (ori_images[:, :, :, 1] * 0.224 + 0.456)
    ori_images[:, :, :, 2] = (ori_images[:, :, :, 2] * 0.225 + 0.406)

    return ori_images.permute(0,3,1,2).contiguous()

def norm(image):
    ori_images = image.permute(0, 2, 3, 1)#.cpu().numpy()
    #orig_images = np.zeros_like(img_temp)
    ori_images[:, :, :, 0] = (ori_images[:, :, :, 0] - 0.485)/ 0.229
    ori_images[:, :, :, 1] = (ori_images[:, :, :, 1] - 0.456)/ 0.224
    ori_images[:, :, :, 2] = (ori_images[:, :, :, 2] - 0.406)/ 0.225

    return ori_images.permute(0,3,1,2).contiguous()



def denormalize_img(imgs=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:,0,:,:] = imgs[:,0,:,:] * std[0] + mean[0]
    _imgs[:,1,:,:] = imgs[:,1,:,:] * std[1] + mean[1]
    _imgs[:,2,:,:] = imgs[:,2,:,:] * std[2] + mean[2]
    _imgs = _imgs.type(torch.uint8)

    return _imgs

def denormalize_img2(imgs=None):
    #_imgs = torch.zeros_like(imgs)
    imgs = denormalize_img(imgs)

    return imgs / 255.0

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--teacher-model', type=str, default='toco',
                        help='model name')  
    parser.add_argument('--student-model', type=str, default='toco',
                        help='model name')                      
    parser.add_argument('--student-backbone', type=str, default='vit_base_patch16_224',
                        help='backbone name')
    parser.add_argument('--teacher-backbone', type=str, default='vit_base_patch16_224',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='voc12',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/VOC2012', 
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[448,448], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=255, metavar='N',
                        help='ignore label')
    
    parser.add_argument("--train_list", default="/opt/data/private/diffkd_segmentation/dataset/voc12/train_aug.txt", type=str)#
    parser.add_argument("--infer_list", default="/opt/data/private/diffkd_segmentation/dataset/voc12/val.txt", type=str)#
    parser.add_argument("--cam_scales", default=(1.0,0.5,), #(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")
    
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--batch-size', type=int, default=6, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=40000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=6e-5, metavar='LR', 
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-2, metavar='M',
                        help='w-decay (default: 5e-4)')

    parser.add_argument("--kd-temperature", type=float, default=1.0, help="logits KD temperature")
    
    parser.add_argument("--lambda-kd", type=float, default=1., help="lambda_kd")
    
    # cuda setting
    parser.add_argument('--gpu-id', type=str, default='1') 
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)#0
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='./work_dirs/dgkd-wlss/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='./work_dirs/dgkd-wlss/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=400,
                        help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=400,#400
                        help='per iters to val')
    parser.add_argument('--teacher-pretrained-base', type=str, default='None',
                        help='pretrained backbone')
    parser.add_argument('--teacher-pretrained', type=str, default='./ckpts/toco_vit-b_voc_20k.pth',#
                        help='pretrained seg model')
    parser.add_argument('--student-pretrained-base', type=str, default='./ckpts/ilsvrc-cls_rna-a1_cls1000_ep-0001.params',#,
                    help='pretrained backbone')
    parser.add_argument('--student-pretrained', type=str, default='None',
                        help='pretrained seg model')
    #parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5), help="multi_scales for cam")
    parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")
    parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")

                        
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    args.aux=True

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

        train_dataset=VOC12ClassificationDataset(args.train_list, voc12_root=args.data,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=448, crop_method="random",aug=True,rescale_range=args.scales)#448
        val_dataset=VOC12ClassificationDatasetMSF(args.infer_list, voc12_root=args.data, scales=args.cam_scales)
    
        args.batch_size = args.batch_size // num_gpus
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iterations)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d

        self.t_model = get_segmentation_model(model=args.teacher_model, 
                                            backbone=args.teacher_backbone,
                                            local_rank=args.local_rank,
                                            pretrained_base=False,
                                            pretrained=args.teacher_pretrained,
                                            test=False, 
                                            norm_layer=nn.BatchNorm2d,
                                            num_class=train_dataset.num_class).to(self.device)

        self.s_model = get_segmentation_model(model=args.student_model, 
                                            backbone=args.student_backbone,
                                            local_rank=args.local_rank,
                                            pretrained_base=True,
                                            pretrained=args.teacher_pretrained, #None,
                                            test=False,  
                                            norm_layer=BatchNorm2d,
                                            num_class=train_dataset.num_class).to(self.device)
        


        
        for t_n, t_p in self.t_model.named_parameters():
            t_p.requires_grad = False
        self.t_model.eval()
        self.s_model.eval()


        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.s_model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        
        self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label).to(self.device)
        self.criterion_kd = DiffKD(train_dataset.num_class)
        self.criterion_kd.cuda()

        self.criterion_kd_cam = DiffKD(train_dataset.num_class-1,kernel_size=3)
        self.criterion_kd_cam.cuda()

        self.criterion_kd_cam_aux = DiffKD(train_dataset.num_class-1,kernel_size=3)
        self.criterion_kd_cam_aux.cuda()

        self.criterion_kd_f = DiffKD(256)#,kernel_size=3
        self.criterion_kd_f.cuda()


        
        # # add kd loss to student
        self.s_model._criterion_kd = self.criterion_kd
        self.s_model._criterion_kd_cam = self.criterion_kd_cam
        self.s_model._criterion_kd_cam_aux = self.criterion_kd_cam_aux
        self.s_model._criterion_kd_f=self.criterion_kd_f
        
        self.kd_loss=DIST().to(self.device)#CriterionKD()
        if self.args.aux:
            self.criterion_kd_f_aux = DiffKD(256)
            self.criterion_kd_f_aux.cuda()
            self.s_model._criterion_kd_f_aux=self.criterion_kd_f_aux

        logger.info(self.s_model)

        
    
        params_list = nn.ModuleList([])
        params_list.append(self.s_model)
    
        param_groups = self.s_model.get_param_groups()

        for param in list(self.s_model._criterion_kd.parameters()):
            param_groups[3].append(param)
        
        for param in list(self.s_model._criterion_kd_f.parameters()):
            param_groups[3].append(param)
        
        for param in list(self.s_model._criterion_kd_f_aux.parameters()):
            param_groups[3].append(param)

        self.optimizer = getattr(optimizer, args.optimizer)(
            params=[
                {
                    "params": param_groups[0],
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": param_groups[1],
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": param_groups[2],
                    "lr": args.lr * 10,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": param_groups[3],
                    "lr": args.lr * 10,
                    "weight_decay": args.weight_decay,
                },
            ],
            lr=args.lr,
            weight_decay=1e-2,
            betas=(0.9, 0.999),
            warmup_iter=1500,
            max_iter=args.max_iterations,
            warmup_ratio=1e-6,
            power=0.9)



        if args.distributed:
            self.s_model = nn.parallel.DistributedDataParallel(self.s_model, 
                                                                device_ids=[args.local_rank],
                                                                output_device=args.local_rank)
            
            
        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0


    def adjust_lr(self, base_lr, iter, max_iter, power):
        cur_lr = 1e-4 + (base_lr - 1e-4)*((1-float(iter)/max_iter)**(power))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

        return cur_lr

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def reduce_mean_tensor(self, tensor):
        rt = tensor.clone()
        #dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.num_gpus
        return rt

    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(args.max_iterations))

        self.s_model.train()
        scaler = GradScaler()
        ncrops = 10
        par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).to(self.device)

        for iteration, (images, d_images,depth_images,targets,label, _,img_box) in enumerate(self.train_loader): #,img_box,crops
            iteration = iteration + 1
            images = images.to(self.device)
            d_images = d_images.to(self.device)
            inputs_denorm = denormalize_img2(d_images.clone())
            depth_images = depth_images.to(self.device)

            targets = targets.long().to(self.device)
           
            label=label.to(self.device)
            b, c, h, w = images.shape
            with torch.no_grad():
                t_cls, t_segs, t_cams,t_cams_aux, t_cls_aux,t_feat1,t_feat2  = self.t_model(images,None,crops=None,n_iter=iteration)
                
            
            with autocast():
                # # get local crops from uncertain regions
                s_cams, s_cams_aux = multi_scale_cam2(self.s_model, inputs=d_images, scales=args.cam_scales)
                s_cls, s_segs, s_cam,s_cam_aux, s_cls_aux,s_feat1,s_feat2  = self.s_model(d_images,depth_images,crops=None,n_iter=iteration)#roi_crops

                # cls loss & aux cls loss
                cls_loss = F.multilabel_soft_margin_loss(s_cls, label)
                cls_loss_aux = F.multilabel_soft_margin_loss(s_cls_aux, label)

                ddim_loss_cam,_,kd_loss_cam=self.criterion_kd_cam(s_cam, t_cams)
                ddim_loss_cam_aux,_,kd_loss_cam_aux=self.criterion_kd_cam(s_cam_aux, t_cams_aux)
                
                ddim_loss, rec_loss, kd_loss =self.criterion_kd(s_segs, t_segs)
                
                ddim_loss_f,rec_loss_f,kd_loss_f=self.criterion_kd_f(s_feat1, t_feat1) #
                
                ddim_loss_f_aux, rec_loss_f_aux, kd_loss_f_aux = self.criterion_kd_f_aux(s_feat2, t_feat2)


                ddim_loss_total=ddim_loss+ddim_loss_f+ddim_loss_f_aux+ddim_loss_cam+ddim_loss_cam_aux
                kd_loss_total=kd_loss+kd_loss_f+kd_loss_f_aux+kd_loss_cam+kd_loss_cam_aux

                # seg_loss & reg_loss
                valid_cam, _ = cam_to_label(s_cams.detach(), cls_label=label, img_box=img_box, ignore_mid=True, bkg_thre=0.35, high_thre=0.6, low_thre=0.15, ignore_index=args.ignore_label)
                refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=label,  high_thre=0.6, low_thre=0.15, ignore_index=args.ignore_label, img_box=img_box, )
                s_segs = F.interpolate(s_segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
                seg_loss = get_seg_loss(s_segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_label)

                print("seg_loss:",seg_loss)


                
                # warmup
                if iteration <= 2000:
                    task_loss= 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.0 * seg_loss
                    losses =task_loss +ddim_loss_total+kd_loss_total 
                else:
                    task_loss= 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.1* seg_loss
                    losses = task_loss+ddim_loss_total+kd_loss_total
               
                
                lr = self.optimizer.param_groups[0]['lr']

                # scale the loss to the mean of the accumulated batch size
                #losses = losses / accumulation_steps

            self.optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(self.optimizer)
            scaler.update()




            task_losses_reduced = self.reduce_mean_tensor(task_loss)
            kd_losses_reduced = self.reduce_mean_tensor(kd_loss)
            kd_losses_reduced_a = self.reduce_mean_tensor(kd_loss_f_aux)
            ddim_losses_reduced = self.reduce_mean_tensor(ddim_loss)
            ddim_losses_reduced_a = self.reduce_mean_tensor(ddim_loss_f_aux)

            ddim_losses_f = self.reduce_mean_tensor(ddim_loss_f)
            kd_losses_f = self.reduce_mean_tensor(kd_loss_f)

            
            eta_seconds = ((time.time() - start_time) / iteration) * (args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} || KD Loss: {:.4f}" \
                        " || DDIM Loss: {:.4f} || KDA Loss: {:.4f} || DDIMA Loss: {:.4f} || Rec Loss: {:.4f} || RecA Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'], task_losses_reduced.item(),
                        kd_losses_reduced.item(), ddim_losses_reduced.item(), kd_losses_reduced_a.item(), ddim_losses_reduced_a.item(), rec_loss.item(), rec_loss_f_aux.item(),# kd_fea_loss.item(),#fd_loss_reduced.item(),#kd_losses_f.item(),#
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.s_model, self.args, is_best=False)#self.enhanced_model,

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.s_model.train()

        save_checkpoint(self.s_model,self.args, is_best=False)#self.f_detector,self.enhanced_model, 
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / args.max_iterations))


    def validation(self):
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.s_model.module
        else:
            model = self.s_model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (images, d_images,depth_images,target,label, filename) in enumerate(self.val_loader):
            image = images[0].to(self.device)
            d_image = d_images[0].to(self.device)
            depth_image = depth_images[0].to(self.device)

            target = target.to(self.device)
            label=label.to(self.device)


            with torch.no_grad():
                
                cls, segs, cam,_, cls_aux,_,_  =model(d_image,depth_image)
               
                
            B, H, W = target.size()
            segs = F.interpolate(segs, (H, W), mode='bilinear', align_corners=True)
            


            target = target.view(H * W)
            eval_index = target != args.ignore_label
            
            full_probs = segs.view(1, self.metric.nclass, H * W)[:,:,eval_index].unsqueeze(-1)
            
            target = target[eval_index].unsqueeze(0).unsqueeze(-1)

            self.metric.update(full_probs, target)
            mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation mIoU: {:.3f}".format(i + 1, mIoU))
        
        if self.num_gpus > 1:
            sum_total_correct = torch.tensor(self.metric.total_correct).cuda().to(args.local_rank)
            sum_total_label = torch.tensor(self.metric.total_label).cuda().to(args.local_rank)
            sum_total_inter = torch.tensor(self.metric.total_inter).cuda().to(args.local_rank)
            sum_total_union = torch.tensor(self.metric.total_union).cuda().to(args.local_rank)
            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            IoU = 1.0 * sum_total_inter / (2.220446049250313e-16 + sum_total_union)
            mIoU = IoU.mean().item()

            logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}".format(mIoU * 100))


        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
            save_checkpoint(self.s_model,self.args, is_best)
        synchronize()


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'kd_{}_{}_{}.pth'.format(args.student_model, args.student_backbone, args.dataset)
    filename = os.path.join(directory, filename)


    if args.distributed:
        model = model.module
    
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'kd_{}_{}_{}_best_model.pth'.format(args.student_model, args.student_backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)



if __name__ == '__main__':
    args = parse_args()

    # reference maskrcnn-benchmark
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = False
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    

    '''fix random seed'''
    seed = 1 #args.seed + args.rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic=True

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.student_model, args.teacher_backbone, args.student_backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
