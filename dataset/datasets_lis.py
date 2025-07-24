import torch
from torch.utils import data
import os.path as osp
import numpy as np
import random
import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

from PIL import Image
import os
from torchvision import transforms
#import torchvision.transforms as standard_transforms

import imageio
from . import imutils

ignore_label=255
id_to_trainid = {1: ignore_label, 2: 1, 3: ignore_label, 4: ignore_label,
                5: 5, 6: 4, 7: 2, 8: ignore_label,
                9: 6, 10: ignore_label, 11: 7, 12: ignore_label, 13: ignore_label,
                14: 3, 15: ignore_label, 16: ignore_label, 17: ignore_label,
                18: ignore_label, 19:ignore_label, 20: 8}


def id2trainId(label, reverse=False):
        label_copy = label.copy()#.astype('int32')
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy






IMG_FOLDER_NAME = 'RGB-normal' 
IMG_FOLDER_NAME2 = "RGB-dark" 
IMG_FOLDER_NAME3='vis_depth_lis'
MASK_FOLDER_NAME = 'RGB-dark'
IGNORE = 255

CAT_LIST = [ 'bicycle',  'car', 'motorbike', 'bus', 'bottle',
        'chair', 'diningtable', 'tvmonitor']

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

cls_labels_dict = np.load('/opt/data/private/DGKD-WLSS/dataset/voc12/cls_labels.npy', allow_pickle=True).item()

def denorm(ori_images):
    #ori_images = image.permute(0, 2, 3, 1)#.cpu().numpy()
    #orig_images = np.zeros_like(img_temp)
    ori_images[:, :, 0] = (ori_images[:, :, 0] * 0.229 + 0.485)*255
    ori_images[:, :, 1] = (ori_images[:, :, 1] * 0.224 + 0.456)*255
    ori_images[:, :, 2] = (ori_images[:, :, 2] * 0.225 + 0.406)*255

    return ori_images.astype(np.uint8)#.permute(0,3,1,2).contiguous()

def norm(ori_images):
    #ori_images = image.permute(0, 2, 3, 1)#.cpu().numpy()
    #orig_images = np.zeros_like(img_temp)
    ori_images[ :, :, 0] = (ori_images[ :, :, 0] - 0.485)/ 0.229
    ori_images[ :, :, 1] = (ori_images[ :, :, 1] - 0.456)/ 0.224
    ori_images[ :, :, 2] = (ori_images[ :, :, 2] - 0.406)/ 0.225

    return ori_images#.permute(0,3,1,2).contiguous()

def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s #s[:4] + '_' + s[4:]


def load_image_label_list_from_npy(img_name_list):

    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])

def get_img_path(img_name, lis_root,img_set):
    if '_' in str(img_name):
        img_path='/opt/data/private/DGKD-WLSS/dataset/VOC2012/JPEGImages'
        return os.path.join(img_path, img_name + '.jpg')
    else:
        img_name=str(int(img_name)-1)
        return os.path.join(lis_root, IMG_FOLDER_NAME,img_set,'imgs', img_name + '.png')

def get_label_path(img_name, lis_root,img_set):
    if '_' in img_name:
        img_path='/opt/data/private/DGKD-WLSS/dataset/VOC2012/SegmentationClassAug'
        return os.path.join(img_path, img_name + '.png')
    else:
        return os.path.join(lis_root, MASK_FOLDER_NAME,img_set, 'masks',img_name + '.png')

def get_img_path2(img_name, lis_root,img_set):
    if '_' in img_name:
        img_path='/opt/data/private/DGKD-WLSS/dataset/dark_VOC2012'
        return os.path.join(img_path, img_name + '.jpg')
    else:
        #img_name=str(int(img_name)+1)  
        return os.path.join(lis_root, IMG_FOLDER_NAME2,img_set,'imgs', img_name + '.JPG')#'.jpg'

def get_img_path3(img_name, lis_root):
    if '_' in img_name:
        img_path='/opt/data/private/DGKD-WLSS/dataset/vis_depth_voc'
        return os.path.join(img_path, img_name + '_depth.png')
    else:
        return os.path.join(lis_root, IMG_FOLDER_NAME3, img_name + '_depth.png')


def load_img_name_list(dataset_path):

    img_name_list = np.loadtxt(dataset_path, dtype=str)

    return img_name_list


class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class LISImageDataset(data.Dataset):

    def __init__(self, img_name_list_path, lis_root,image_set,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):
    #def __init__(self,lis_root, img_name_list_path, transform=None):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.lis_root = lis_root
        self.img_set=image_set

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch
        self.num_class =9

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str=str(name)

        img = np.asarray(imageio.imread(get_img_path(name_str, self.lis_root,self.img_set)))#str(int(name_str)-1)
        d_img=np.asarray(imageio.imread(get_img_path2(name_str, self.lis_root,self.img_set)))
        depth_img=np.asarray(imageio.imread(get_img_path3(name_str, self.lis_root)))
        mask=np.asarray(imageio.imread(get_label_path(name_str, self.lis_root,self.img_set)))
        
        if '_' in name_str:
            mask=id2trainId(mask)
        
        if self.resize_long:
            img,d_img,depth_img,mask = imutils.random_resize_long3(img,d_img,depth_img,mask, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img,d_img,depth_img,mask = imutils.random_scale3(img,d_img,depth_img,mask, scale_range=self.rescale, order=3)
            
        
        if self.img_normal:
            img = self.img_normal(img)
            d_img = self.img_normal(d_img)
            depth_img = self.img_normal(depth_img)
        

        if self.hor_flip:
            img,d_img,depth_img,mask = imutils.random_lr_flip3(img,d_img,depth_img,mask)

        if self.crop_size:
            if self.crop_method == "random":
                img,d_img,depth_img,mask = imutils.random_crop3(img,d_img,depth_img,mask, self.crop_size, 0)   
            else:
                img,d_img,depth_img,mask = imutils.top_left_crop3(img,d_img,depth_img,mask, self.crop_size, 0)


        if self.to_torch:
            img = imutils.HWC_to_CHW(img)
            d_img = imutils.HWC_to_CHW(d_img)
            depth_img=imutils.HWC_to_CHW(depth_img)

        return {'name': name_str, 'img': img,'d_img':d_img,'depth_img':depth_img,'target':mask}#  'depth_img':depth_img,
        

class LISClassificationDataset(LISImageDataset):

    def __init__(self, img_name_list_path, lis_root, image_set,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, lis_root,image_set,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        label_idx=torch.unique(torch.from_numpy(out['target']))
        label=torch.zeros(9)
        for id in label_idx:
            if id!=255:
                label[int(id)]=1
        out['label'] = label
        return out['img'],out['d_img'],out['depth_img'],out['target'],out['label'],out['name'] #out['hist_img'] out['depth_img']



class LISClassificationDatasetMSF(LISClassificationDataset):
    def __init__(self, img_name_list_path, lis_root, image_set,img_normal=TorchvisionNormalize(), scales=(1.0,)):
        #self.scales = scales
        super().__init__(img_name_list_path, lis_root,image_set, img_normal=img_normal)
        self.scales = scales
        self.img_set=image_set
       

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str=str(name)

        img = imageio.imread(get_img_path(name_str, self.lis_root,self.img_set))#str(int(name_str)-1)
        d_img=imageio.imread(get_img_path2(name_str, self.lis_root,self.img_set))
        depth_img=imageio.imread(get_img_path3(name_str, self.lis_root))
        mask=imageio.imread(get_label_path(name_str, self.lis_root,self.img_set))
        

        ms_img_list = []
        d_ms_img_list=[]
        depth_ms_img_list=[]
        hist_ms_img_list=[]
        mask_ms_list=[]
        

        if '_' in name_str:
            mask=id2trainId(np.asarray(mask))

        s_mask=imutils.pil_rescale(np.asarray(mask), 1, order=0)
        label_idx=torch.unique(torch.from_numpy(s_mask))
        label=torch.zeros(9)
        for j in label_idx:
            if j!=255:
                label[int(j)]=1
        for s in self.scales:
            if s == 1:
                s_img = img
                s_d_img=d_img
                s_depth_img=depth_img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
                s_d_img = imutils.pil_rescale(d_img, s, order=3)
                s_depth_img = imutils.pil_rescale(depth_img, s, order=3)
            

            s_img = self.img_normal(s_img)
            s_d_img=self.img_normal(s_d_img)
            s_depth_img=self.img_normal(s_depth_img)
            

            s_img = imutils.HWC_to_CHW(s_img)
            s_d_img=imutils.HWC_to_CHW(s_d_img)
            s_depth_img=imutils.HWC_to_CHW(s_depth_img)
            
            ms_img_list.append(s_img)  
            depth_ms_img_list.append(s_depth_img)
            d_ms_img_list.append(s_d_img) 

       
        out = {"name": name_str, "img": ms_img_list,"d_img": d_ms_img_list,"depth_img":depth_ms_img_list, "target":s_mask,
               "label": label}
        return out['img'],out['d_img'],out['depth_img'],out['target'],out['label'], out['name']
