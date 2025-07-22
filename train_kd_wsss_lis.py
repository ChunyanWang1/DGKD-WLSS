import argparse
import time
import datetime
import os
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

from utils.distributed import *
from utils.logger import setup_logger
from utils.score import SegmentationMetric
from dataset.datasets_lis import LISClassificationDataset,LISClassificationDatasetMSF # CSTrainValSet,
from utils.flops import cal_multi_adds, cal_param_size#,PolynomialLR
#from models.retinex_network import Retinex_decom
# from models import dip_origin

from losses.dist_kd import DIST
# from losses.mmdloss import MMDLoss

# from utils.cutmix import obtain_bbox,mix

#from losses.flaw_detector import FlawDetector,FDGTGenerator,FlawDetectorCriterion,FlawmapHandler,DCGTGenerator
from torch.cuda.amp import autocast, GradScaler

ignore_label=255
id_to_trainid = {1: ignore_label, 2: 1, 3: ignore_label, 4: ignore_label,
                5: 5, 6: 4, 7: 2, 8: ignore_label,
                9: 6, 10: ignore_label, 11: 7, 12: ignore_label, 13: ignore_label,
                14: 3, 15: ignore_label, 16: ignore_label, 17: ignore_label,
                18: ignore_label, 19:ignore_label, 20: 8}

# ignore_label=0
# id_to_trainid = {1: ignore_label, 2: 1, 3: ignore_label, 4: ignore_label,
#                 5: 5, 6: 4, 7: 2, 8: ignore_label,
#                 9: 6, 10: ignore_label, 11: 7, 12: ignore_label, 13: ignore_label,
#                 14: 3, 15: ignore_label, 16: ignore_label, 17: ignore_label,
#                 18: ignore_label, 19:ignore_label, 20: 8}

voc2lis_index=[0,2,7,14,6,5,9,11,20]


def id2trainId(label, reverse=False):
        label_copy = label.clone()#.astype('int32')
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy



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

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--teacher-model', type=str, default='ssss',
                        help='model name')  
    parser.add_argument('--student-model', type=str, default='ssss',
                        help='model name')                      
    parser.add_argument('--student-backbone', type=str, default='resnet38',
                        help='backbone name')
    parser.add_argument('--teacher-backbone', type=str, default='resnet38',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='LIS',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/LIS',  
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[321,321], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=255, metavar='N',
                        help='ignore label')
    
    parser.add_argument("--train_list", default="./dataset/LIS/train_dark.txt", type=str)
    parser.add_argument("--infer_list", default="./dataset/LIS/test_dark.txt", type=str)
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
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M',
                        help='w-decay (default: 5e-4)')

    parser.add_argument("--kd-temperature", type=float, default=1.0, help="logits KD temperature")
    
    parser.add_argument("--lambda-kd", type=float, default=1., help="lambda_kd")
    
    # cuda setting
    parser.add_argument('--gpu-id', type=str, default='0') 
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
    parser.add_argument('--val-per-iters', type=int, default=400,
                        help='per iters to val')
    parser.add_argument('--teacher-pretrained-base', type=str, default='None',
                        help='pretrained backbone')
    parser.add_argument('--teacher-pretrained', type=str, default='./ckpts/model_lis_normal.pth',
                        help='pretrained seg model')
    parser.add_argument('--student-pretrained-base', type=str, default='./ckpts/ilsvrc-cls_rna-a1_cls1000_ep-0001.params',
                    help='pretrained backbone')
    parser.add_argument('--student-pretrained', type=str, default='None',
                        help='pretrained seg model')

                        
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

    if args.student_backbone.startswith('resnet'):
        args.aux = True
    elif args.student_backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

        train_dataset=LISClassificationDataset(args.train_list, lis_root=args.data,image_set='train',
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=321, crop_method="random")#448
        val_dataset=LISClassificationDatasetMSF(args.infer_list, lis_root=args.data,image_set='test',scales=args.cam_scales)
    
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
                                            pretrained_base='None',
                                            pretrained=args.teacher_pretrained,
                                            aux=True, 
                                            norm_layer=nn.BatchNorm2d,
                                            num_class=train_dataset.num_class).to(self.args.local_rank)

        self.s_model = get_segmentation_model(model=args.student_model, 
                                            backbone=args.student_backbone,
                                            local_rank=args.local_rank,
                                            pretrained_base=args.student_pretrained_base,
                                            pretrained='None',
                                            aux=args.aux, 
                                            norm_layer=BatchNorm2d,
                                            num_class=train_dataset.num_class).to(self.device) #train_dataset.num_class
        
        
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

        self.criterion_kd_f = DiffKD(256)
        self.criterion_kd_f.cuda()

        
        # # add kd loss to student
        self.s_model._criterion_kd = self.criterion_kd
        self.s_model._criterion_kd_f=self.criterion_kd_f
        
        self.kd_loss=DIST().to(self.device)#CriterionKD()
        if self.args.aux:
            self.criterion_kd_aux = DiffKD(train_dataset.num_class)#self.s_model, self.t_model,args.student_backbone,args.teacher_backbone,
            self.criterion_kd_aux.cuda()
            self.s_model._criterion_kd_aux = self.criterion_kd_aux

            self.criterion_kd_f_aux = DiffKD(256)#,kernel_size=3
            self.criterion_kd_f_aux.cuda()
            self.s_model._criterion_kd_f_aux=self.criterion_kd_f_aux

        logger.info(self.s_model)
    
        params_list = nn.ModuleList([])
        params_list.append(self.s_model)

        self.optimizer = torch.optim.SGD(params_list.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)



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
        for iteration, (images, d_images,depth_images,targets,label, _) in enumerate(self.train_loader):
            iteration = iteration + 1

            
            
            images = images.to(self.device)
            d_images = d_images.to(self.device)
            depth_images = depth_images.to(self.device)
            targets = targets.long().to(self.device)
           
            label=label.to(self.device)
            
            with torch.no_grad():
                t_outputs = self.t_model(images,None,label[:,1:])
            with autocast():
                s_outputs = self.s_model(d_images,depth_images,label[:,1:]) 
                ddim_loss, kd_loss =self.criterion_kd(s_outputs[2], t_outputs[2])
                
                ddim_loss_f,kd_loss_f=self.criterion_kd_f(s_outputs[3], t_outputs[3]) #
                ddim_loss_f_aux, kd_loss_f_aux = self.criterion_kd_f_aux(s_outputs[4], t_outputs[4])
                
                if iteration<=8000:
                    task_loss = F.multilabel_soft_margin_loss(s_outputs[0], label[:,1:])#+s_outputs[1].mean()#self.criterion_cls(s_outputs[0], label).mean()
                else:
                    task_loss = F.multilabel_soft_margin_loss(s_outputs[0], label[:,1:])+s_outputs[1].mean()

                losses = task_loss+ kd_loss + ddim_loss +ddim_loss_f+kd_loss_f+ddim_loss_f_aux+kd_loss_f_aux
                
                lr = self.adjust_lr(base_lr=args.lr, iter=iteration-1, max_iter=args.max_iterations, power=0.9)

                # scale the loss to the mean of the accumulated batch size
                #losses = losses / accumulation_steps

            self.optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(self.optimizer)
            scaler.update()


            task_losses_reduced = self.reduce_mean_tensor(task_loss)
            kd_losses_reduced = self.reduce_mean_tensor(kd_loss)
            ddim_losses_reduced = self.reduce_mean_tensor(ddim_loss)
            ddim_losses_f = self.reduce_mean_tensor(ddim_loss_f)
            kd_losses_f = self.reduce_mean_tensor(kd_loss_f)

            ddim_losses_f_aux = self.reduce_mean_tensor(ddim_loss_f_aux)
            kd_losses_f_aux = self.reduce_mean_tensor(kd_loss_f_aux)

            
            
            eta_seconds = ((time.time() - start_time) / iteration) * (args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} || KD Loss: {:.4f}" \
                        " || DDIM Loss: {:.4f} || kd_fea Loss: {:.4f} || ddim_fea Loss: {:.4f}|| kd_fea_aux Loss: {:.4f} || ddim_fea_aux Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'], task_losses_reduced.item(),
                        kd_losses_reduced.item(), ddim_losses_reduced.item(),  kd_losses_f.item(),ddim_losses_f.item(),kd_losses_f_aux.item(),ddim_losses_f_aux.item(), 
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

            d_image2 = d_images[1].to(self.device)
            depth_image2 = depth_images[1].to(self.device)


            with torch.no_grad():
                outputs = model(d_image,depth_image,None)#d_enhanced_img,,student='test' d_image
                outputs2 = model(d_image2,depth_image2,None)
                

            B, _,H, W = d_image.size()
            outputs[2] = F.interpolate(outputs[2], (H, W), mode='bilinear', align_corners=True)
            outputs2[2] = F.interpolate(outputs2[2], (H, W), mode='bilinear', align_corners=True)
            outputs[2]=(outputs[2]+outputs2[2])/2


            target = target.view(H * W)
            eval_index = target != args.ignore_label
            
            full_probs = outputs[2].view(1, self.metric.nclass, H * W)[:,:,eval_index].unsqueeze(-1)
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

            logger.info("Overall validation mIoU: {:.3f}".format(mIoU * 100))


        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
            save_checkpoint(self.s_model,self.args, is_best)# self.enhanced_model,
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
